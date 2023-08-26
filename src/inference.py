import os
import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict
from typing import Any, Dict
import numpy as np
import pandas as pd
import torch

from src.data_io import load_dicom_series
from src.segmentation.model import load_models as seg_load_models
from src.segmentation.trainer import inference as seg_inference
from src.image_processing import (
    apply_preprocess,
    crop_organ,
    kidney_split,
    apply_postprocess,
    kidney_specific,
    resize_3d,
    resize_1d,
)
from src.classification.model import load_models as cls_load_models
from src.classification.trainer import inference as cls_inference

# organs dict (for SEG and LSK models)
organ_index_dict_inv = {
    0: 'liver',
    1: 'spleen',
    2: 'kidney',
    3: 'bowel'
}
organ_index_dict = {v: k for k, v in organ_index_dict_inv.items()}

# labels dict (for BE models)
label_index_dict_inv = {
    0: 'bowel',
    1: 'extravasation'
}

class Inference:
    """推論パイプライン.
    Note:
        - details in exp012.ipynb
    """

    def __init__(self, CFG_INF: Any, CFG_SEG: Any, CFG_LSK: Any, CFG_BE: Any):
        self.CFG_INF = CFG_INF
        self.CFG_SEG = CFG_SEG
        self.CFG_LSK = CFG_LSK
        self.CFG_BE = CFG_BE

        self.seg_models = seg_load_models(CFG_SEG, mode=self.CFG_INF.seg_model_mode)
        self.lsk_models = cls_load_models(CFG_LSK, mode=self.CFG_INF.lsk_model_mode)
        self.be_models = cls_load_models(CFG_BE, mode=self.CFG_INF.be_model_mode, framework="timm")

    def __call__(self, pid: int) -> tuple:
        """inference process.
        1. load images from dicom files.
        2. create segmentation masks.
        3. create liver, spleen, kidney volumes.
        4. inference lsk models.
        5. inference be models.
        Args:
            pid (int): patient id.
        Return example:
            dict: {
            'pid': 0,
            'bowel_healthy': 0.0,
            'bowel_injury': 0.0,
            'extravasation_healthy': 0.0,
            'extravasation_injury': 0.0,
            'kidney_healthy': 0.0,
            'kidney_low': 0.0,
            'kidney_high': 0.0,
            'liver_healthy': 0.0,
            'liver_low': 0.0,
            'liver_high': 0.0,
            'spleen_healthy': 0.0,
            'spleen_low': 0.0,
            'spleen_high': 0.0
            }
        Note:
            - １症例に複数シリーズ存在する場合、各シリーズに対して推論を行い、全予測結果の最大値を採用する.
            - 推論時間的に厳しければ、最初のシリーズのみを採用するなど検討.
        """
        df_study = self.CFG_INF.df_series_meta[self.CFG_INF.df_series_meta['patient_id']==pid].reset_index(drop=True)
        # df_study内のそれぞれのシリーズを取得して、画像枚数に対して降順にソート.
        df_study = self.get_slices_and_sort(df_study)
        preds = defaultdict(list)
        for sid in df_study["series_id"].to_list()[: self.CFG_INF.max_series]:
            data = self.load_data(pid, sid)
            if data is None:
                continue
            lsk_preds = self.lsk_prediction(data)
            be_preds = self.be_prediction(data)

            for idx, organ in organ_index_dict_inv.items():
                if idx == 3:
                    continue
                preds[organ].append(lsk_preds[idx])
            for idx, label in label_index_dict_inv.items():
                pred = np.array([be_preds[idx]])
                preds[label].append(pred)

        ret = {"patient_id": pid}
        for k, v in preds.items():
            v = np.array(v)
            ret[k] = np.max(v, axis=0)
        ret = self.convert_submission_format(ret)
        return ret

    def load_data(self, pid: int, sid: int) -> np.ndarray:
        """dicomから画像を読み込む.
        Args:
            pid (int): patient id.
            sid (int): series id.
        Returns:
            np.ndarray: (Z, H, W) normalized CT series.
        Note:
            - preprocessは全モデル共通なので、ここで行う.
            - H, Wはすべてself.CFG_INF.image_sizeにresizeされる.
        """
        series_path = os.path.join(self.CFG_INF.image_dir, str(pid), str(sid))
        # sample submissionでこういう例が存在する.
        if not os.path.exists(series_path):
            return None
        image_arr, path_list, meta_list = load_dicom_series(
            series_path, self.CFG_INF.max_slices
        )
        image_arr = apply_preprocess(image_arr, resize=self.CFG_INF.image_size)
        # sample submission対応
        if len(image_arr) < self.CFG_INF.min_slices:
            image_arr = resize_1d(image_arr, self.CFG_INF.min_slices, axis=0)
        return image_arr
    
    def get_slices_and_sort(self, df_study: pd.DataFrame)-> pd.DataFrame:
        """シリーズのスライス数を取得して、スライス数に対して降順にソートする.
        Args:
            df_study (pd.DataFrame): series meta dataframe.
        Returns:
            pd.DataFrame: sorted series meta dataframe.
        """
        pid = df_study['patient_id'][0]
        df_study['n_slices'] = 0
        for i in range(len(df_study)):
            sid = df_study['series_id'][i]
            series_path = os.path.join(self.CFG_INF.image_dir, str(pid), str(sid))
            if os.path.exists(series_path):
                df_study['n_slices'][i] = len(os.listdir(series_path))
        df_study = df_study.sort_values(by='n_slices', ascending=False)
        return df_study

    def lsk_prediction(self, data: np.ndarray) -> np.ndarray:
        """liver, spleen, kidneyの予測値を返す.
        Args:
            data: (Z, H, W).
        Returns:
            np.ndarray: (organs, grades).
        """
        volumes = self.get_lsk_volumes(data)  # (organs, z, h, w)
        lsk_iterator = self.pseudo_iterator(self.CFG_LSK, volumes)
        pred = cls_inference(self.CFG_LSK, self.lsk_models, lsk_iterator)
        return pred

    def get_lsk_volumes(self, data: np.ndarray) -> np.ndarray:
        """Segmentationからliver, spleen, kidneyのvolume dataを作成.
        Args:
            data: (Z, H, W).
        Returns:
            np.ndarray: (organs, z, h, w).
        Note:
            - organsはliver, spleen, kidneyの順番.
            - この関数内でCFG.LSK.image_sizeのreshapeまで行う.
            - 腎臓は左右を分離してからくっつけ直すという特殊な処理が必要.
        """
        masks = self.get_segmentation(data)
        masks = apply_postprocess(self.CFG_SEG, masks)
        arr = []
        for idx, organ in organ_index_dict_inv.items():
            if idx == 3:
                continue
            organ_segment = masks[..., idx]
            if organ_segment.sum() == 0:
                arr.append(np.zeros(self.CFG_LSK.image_size))
                continue
            img_cropped, mask_cropped = crop_organ(data, organ_segment)
            if organ == "kidney":
                kidney_r, kidney_l = kidney_split(img_cropped, mask_cropped)
                img_cropped = kidney_specific(self.CFG_LSK, kidney_r, kidney_l)
            else:
                img_cropped = resize_3d(img_cropped, self.CFG_LSK.image_size)
            arr.append(img_cropped)
        arr = np.stack(arr, axis=0)
        return arr

    def get_segmentation(self, data: np.ndarray) -> np.ndarray:
        """Segmentation modelを使って、各臓器のマスクを作成.
        Args:
            data: (Z, H, W).
        Returns:
            mask: (z, h, w, ch) binarized."""
        seg_iterator = self.pseudo_iterator(self.CFG_SEG, data)
        pred = seg_inference(self.CFG_SEG, self.seg_models, seg_iterator)
        pred = (pred > 0.5).astype(np.uint8)
        return pred

    def be_prediction(self, data: np.ndarray) -> np.ndarray:
        """bowel_injury及びextravasation_injuryの予測を行う.
        Args:
            data: (Z, H, W).
        Returns:
            np.ndarray: [bowel_injury_pred, extravasation_injury_pred].
            example: [0.1, 0.9].
        """
        be_iterator = self.pseudo_iterator(self.CFG_BE, data)
        pred = cls_inference(self.CFG_BE, self.be_models, be_iterator)
        pred = self.be_prediction_postprocess(pred)
        return pred

    def be_prediction_postprocess(self, pred: np.ndarray, p: int=98) -> np.ndarray:
        """スライスごとの予測をシリーズの予測に変換する.
        Args:
            pred: (len(data),['bowel_injury', 'extravasation_injury']).
            p: percentile.
        Returns:
            np.ndarray: ['bowel_injury', 'extravasation_injury'].
        Note:
            - 予測値の最大値から外れ値を考慮したp percentileを採用する.
        """
        bowel = pred[:, 0]
        extravasation = pred[:, 1]
        bowel = np.percentile(bowel, p)
        extravasation = np.percentile(extravasation, p)
        return np.array([bowel, extravasation])

    def pseudo_iterator(self, CFG: Any, images: np.ndarray)-> tuple:
        """evaluation iterator.
        Args:
            CFG: config.
            images: (batch dim, H, W) or (batch dim, Z, H, W).
        """
        batch = CFG.batch_size
        length = len(images)
        arr = []
        if not CFG.expand_ch_dim:
            images = self.add_dummy_array(CFG, images)
        for i in range(length):
            if CFG.expand_ch_dim:
                img = images[i]
                img = img[np.newaxis, ...]
            else:
                img = images[i:i+CFG.n_ch]
            arr.append(img)
            if i != 0 and (i%batch==0 or i == length-1):
                arr = np.stack(arr, axis=0)
                arr = torch.from_numpy(arr.astype(arr.dtype, copy=False))
                yield arr
                arr = []

    def add_dummy_array(self, CFG: Any, images: np.ndarray)-> np.ndarray:
        """chが複数ある場合に、事前に0配列を追加しておく."""
        add_ch = CFG.n_ch//2
        arr = []
        img = np.zeros_like(images[0])
        for i in range(add_ch):
            arr.append(img)
        arr.extend(images)
        for i in range(add_ch):
            arr.append(img)
        arr = np.stack(arr, axis=0)
        return arr

    def convert_submission_format(self, pred: dict) -> dict:
        """提出形式に変換する."""
        converted = dict()
        for idx, organ in organ_index_dict_inv.items():
            if idx == 3:
                continue
            for idx, grade in enumerate(["healthy", "low", "high"]):
                converted[f"{organ}_{grade}"] = pred[organ][idx]
        for idx, label in label_index_dict_inv.items():
            converted[f"{label}_healthy"] = 1 - pred[label][0]
            converted[f"{label}_injury"] = pred[label][0]

        converted["patient_id"] = pred["patient_id"]
        return converted
