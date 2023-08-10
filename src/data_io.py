"""データ入出力関連."""
import os
from typing import Any, List

import numpy as np
import pydicom

from src.dicom_io.meta_processing import get_dtype
from src.dicom_io.typemanager import pixelspacing_type

# 読み込みパラメータの設定.
# 共通データ
common_info = [((0x0008, 0x0060), "Modality", str)]
# 画像として認識するモダリティタグ
image_modality = ["CT", "MR"]
# 画像データから抽出するタグ一覧
# Todo: dataclassに変更
image_info = [
    ((0x0028, 0x0100), "BitsAllocated", int),
    ((0x0028, 0x0103), "PixelRepresentation", int),
    ((0x0028, 0x0030), "PixelSpacing", pixelspacing_type),
    ((0x0028, 0x1052), "RescaleIntercept", int),
    ((0x0028, 0x0010), "Rows", int),
    ((0x0028, 0x0011), "Columns", int),
    ((0x0020, 0x0013), "InstanceNumber", int),
]
# RTSTRUCTデータから抽出するタグ一覧
rs_info = []
info = {
    "common_info": common_info,
    "image_info": image_info,
    "rs_info": rs_info,
}

def load_image_from_dicom(dicom: pydicom.dataset.FileDataset, meta: dict) -> np.ndarray:
    """load image from dicom path."""
    image = dicom.pixel_array.astype(np.int16)
    image += meta["RescaleIntercept"]
    return image

def load_metadata_from_dicom(dicom: pydicom.dataset.FileDataset) -> dict:
    """load metadata from dicom path.

    Note:
        - 現状では、"Modality"が"CT"か"MR"の場合画像として扱う。(image_modality)
        - それ以外の場合は"RTSTRUCT"として扱うが、実際はRTSTRUCTでなくても対応できる想定。
    """
    ret = dict()

    # common info
    for tag, name, type_ in info["common_info"]:
        # RSNA2023のデータは、Modalityが消されている.
        # CTデータとして認識(RSNA2023特異的な処理)
        ret[name] = type_("CT")

    # image info or RTSTRUCT info
    if ret["Modality"] in image_modality:  # image info
        for tag, name, type_ in info["image_info"]:
            # Todo: ごちゃごちゃし始めているので、dataclassに切り分けて整理.
            if tag not in dicom:
                if name == "RescaleIntercept":
                    ret[name] = 0
                    continue
                else:
                    raise Exception(f"no tag: {tag}")
            ret[name] = type_(dicom[tag].value)
    else:  # RTSTRUCT
        for tag, name, type_ in info["rs_info"]:
            ret[name] = type_(dicom[tag].value)

    return ret

def load_dicom_series(dir_: str) -> Any:
    """load dicom series from directory.

    Returns:
        dicom_list: dicomファイルのリスト
        image_arr: 画像データのnumpy配列
        path_list: dicomファイルのパスのリスト
        meta_list: dicomファイルのメタデータのリスト

        dicom,path,metaのリストは長さ同じだが、image_arrは、画像がある分だけ。
        それぞれのリストはスライス順にソートされており、RTSTRUCTなどの、
        画像以外のデータはリストの最後尾に格納されている。

    Note:
        - メモリ節約のため、dicomファイルのpixel_arrayは削除。
    """
    tmp_list = []
    for path in os.listdir(dir_):
        # dicomファイル以外は除外.
        if path[-4:] != ".dcm":
            continue
        dicom = pydicom.read_file(os.path.join(dir_, path))

        meta = load_metadata_from_dicom(dicom)

        # dicomの画像データを削除
        if meta["Modality"] in image_modality:
            image = load_image_from_dicom(dicom, meta)
            del dicom.PixelData
        else:
            image = None

        # 画像データでないならば最後尾に格納.
        index = (
            10**9
            if meta["Modality"] not in image_modality
            else meta["InstanceNumber"]
        )

        tmp_list.append((index, path, dicom, meta, image))

    tmp_list.sort()

    path_list = []
    dicom_list = []
    meta_list = []
    image_arr = []
    # 今回dicomはいらないので省略
    for index, path, dicom, meta, image in tmp_list:
        path_list.append(path)
        # dicom_list.append(dicom)
        meta_list.append(meta)
        if image is not None:
            image_arr.append(image)

    # return dicom_list, np.array(image_arr), path_list, meta_list
    return np.array(image_arr), path_list, meta_list