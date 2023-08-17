import os
from typing import Any, Callable, Tuple, Union

import cv2
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_image(path: str) -> np.ndarray:
    """画像の読み込み.
    Args:
        path (str): 画像のパス.
    Returns:
        numpy.ndarray: 画像.
    Note:
        現在読み込む画像の形式は.png, .npy, .npzのみ対応.
        cv2.IMREAD_UNCHANGED: 16bit画像やアルファチャンネルを考慮した読み込み.
    """
    if path.endswith(".png"):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif path.endswith(".npy"):
        image = np.load(path)
    elif path.endswith(".npz"):
        image = np.load(path)["arr_0"]
    else:
        raise Exception(f"unexpected image format: {path}")
    return image

def resize_1d(image: np.ndarray, imsize: int, axis: int = 2) -> np.ndarray:
    """3次元配列のうち、axisに指定した1次元をリサイズする."""
    # 指定された軸を最後に移動
    image_moved = np.moveaxis(image, axis, -1)
    
    x_old = np.linspace(0, 1, image_moved.shape[-1])
    x_new = np.linspace(0, 1, imsize)
    interpolator = interp1d(x_old, image_moved, axis=-1)

    result_moved = interpolator(x_new)
    # 元の軸の順序に戻す
    result = np.moveaxis(result_moved, -1, axis)
    
    return result

def resize(image: np.ndarray, imsize: Tuple[int, int, int]) -> np.ndarray:
    """ボリュームデータのリサイズ.
    Args:
        image (numpy.ndarray): volume(h, w, z).
        imsize (tuple): リサイズ後の画像サイズ.
    Returns:
        numpy.ndarray: resized volume.
    """
    image = cv2.resize(image, imsize[:2], interpolation=cv2.INTER_LINEAR)
    image = resize_1d(image, imsize[2], axis=2)
    return image


def img2tensor(img, dtype: np.dtype = np.float32):
    """numpy.ndarrayをtorch.tensorに変換する."""
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

def get_label(df: pd.DataFrame, idx: int) -> torch.tensor:
    """ラベルを取得する.healthy, low, highの順"""
    label = []
    for i in ["healthy", "low", "high"]:
        label.append(df[i][idx])
    return torch.tensor(label, dtype=torch.float32)


class TrainDatasetSolidOrgans(Dataset):
    """固形臓器(liver, spleen, kidney)の学習用データセット.
    症例ごとのラベルしか持たないため、ボリュームデータを返す.
    """

    def __init__(
        self,
        CFG: Any,
        df: pd.DataFrame,
        preprocess: Union[None, Callable] = None,
        tfms: Union[None, Callable] = None,
    ) -> None:
        self.df = df
        self.CFG = CFG
        self.preprocess = preprocess
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """画像とラベル(mask)の取得.
        Args:
            idx (int): self.dfに対応するデータのインデックス.
        Returns:
            tuple (torch.tensor, torch.tensor): 画像とラベル.
        Note:
            現在読み込む画像の形式は.png, .npy, .npzのみ対応.
        """
        impath = self.df["image_path"][idx]

        # ファイル名がkidney.npyならば左右を結合して読み込む
        if os.path.basename(impath) == "kidney.npy":
            return self.kidney_specific(idx)

        if os.path.exists(impath):
            image = load_image(impath)
        else:
            image = np.zeros(self.CFG.image_size)
        
        # dataset002のボリュームデータは(z, h, w)
        image = image.transpose(1, 2, 0)

        if self.preprocess:
            image = self.preprocess(image)

        image = resize(image, self.CFG.image_size)

        if self.tfms:
            res = self.tfms(image=image)
            image = res["image"]

        if self.CFG.expand_ch_dim:
            # ch as channel
            # (h, w, z) -> (z, h, w) -> (ch, z, h, w)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image.astype(np.float32, copy=False))

        else:
            # z as channel
            # (h, w, z) -> (z, h, w)
            image = img2tensor(image)
            

        label = get_label(self.df, idx)

        return image, label
    
    def kidney_specific(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """左右ラベルのついた腎臓を読み込み、W方向にconcatして返す."""
        impath = self.df["image_path"][idx]
        # 解剖学的な左右を、画像上の左右に置き換えて読み込み
        l, r = impath.replace("kidney.npy", "kidney_r.npy"), impath.replace("kidney.npy", "kidney_l.npy")
        if os.path.exists(l):
            l = load_image(l)
        else:
            l = np.zeros(self.CFG.image_size)
        if os.path.exists(r):
            r = load_image(r)
        else:
            r = np.zeros(self.CFG.image_size)
        # z, h, w -> h, w, z
        l = l.transpose(1, 2, 0)
        r = r.transpose(1, 2, 0)
        if self.preprocess:
            l = self.preprocess(l)
            r = self.preprocess(r)
        # z
        l = resize_1d(l, self.CFG.image_size[0] ,axis=2)
        r = resize_1d(r, self.CFG.image_size[0] ,axis=2)
        # h
        l = resize_1d(l, self.CFG.image_size[1] ,axis=0)
        r = resize_1d(r, self.CFG.image_size[1] ,axis=0)
        # w
        image = np.concatenate([l, r], axis=1)
        image = resize_1d(image, self.CFG.image_size[2] ,axis=1)

        if self.tfms:
            res = self.tfms(image=image)
            image = res["image"]
        if self.CFG.expand_ch_dim:
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image.astype(np.float32, copy=False))
        else:
            image = img2tensor(image)
        label = get_label(self.df, idx)
        return image, label



class TestDatasetSolidOrgans(Dataset):
    """固形臓器(liver, spleen, kidney)のテスト用データセット."""

    def __init__(
        self,
        CFG: Any,
        df: pd.DataFrame,
        preprocess: Union[None, Callable] = None,
        tfms: Union[None, Callable] = None,
    ) -> None:
        self.df = df
        self.CFG = CFG
        self.preprocess = preprocess
        self.tfms = tfms

        assert self.tfms is None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """画像とラベル(mask)の取得.
        Args:
            idx (int): self.dfに対応するデータのインデックス.
        Returns:
            tuple (torch.tensor, torch.tensor): 画像とラベル.
        """
        impath = self.df["image_path"][idx]

        # ファイル名がkidney.npyならば左右を結合して読み込む
        if os.path.basename(impath) == "kidney.npy":
            return self.kidney_specific(idx)

        if os.path.exists(impath):
            image = load_image(impath)
        else:
            image = np.zeros(self.CFG.image_size)
        
        # dataset002のボリュームデータは(z, h, w)
        image = image.transpose(1, 2, 0)

        if self.preprocess:
            image = self.preprocess(image)

        image = resize(image, self.CFG.image_size)

        if self.tfms:
            res = self.tfms(image=image)
            image = res["image"]

        if self.CFG.expand_ch_dim:
            # ch as channel
            # (h, w, z) -> (z, h, w) -> (ch, z, h, w)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image.astype(np.float32, copy=False))

        else:
            # z as channel
            # (h, w, z) -> (z, h, w)
            image = img2tensor(image)
            

        label = get_label(self.df, idx)

        return image, label
    
    def kidney_specific(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """左右ラベルのついた腎臓を読み込み、W方向にconcatして返す."""
        impath = self.df["image_path"][idx]
        # 解剖学的な左右を、画像上の左右に置き換えて読み込み
        l, r = impath.replace("kidney.npy", "kidney_r.npy"), impath.replace("kidney.npy", "kidney_l.npy")
        if os.path.exists(l):
            l = load_image(l)
        else:
            l = np.zeros(self.CFG.image_size)
        if os.path.exists(r):
            r = load_image(r)
        else:
            r = np.zeros(self.CFG.image_size)
        # z, h, w -> h, w, z
        l = l.transpose(1, 2, 0)
        r = r.transpose(1, 2, 0)
        if self.preprocess:
            l = self.preprocess(l)
            r = self.preprocess(r)
        # z
        l = resize_1d(l, self.CFG.image_size[0] ,axis=2)
        r = resize_1d(r, self.CFG.image_size[0] ,axis=2)
        # h
        l = resize_1d(l, self.CFG.image_size[1] ,axis=0)
        r = resize_1d(r, self.CFG.image_size[1] ,axis=0)
        # w
        image = np.concatenate([l, r], axis=1)
        image = resize_1d(image, self.CFG.image_size[2] ,axis=1)

        if self.tfms:
            res = self.tfms(image=image)
            image = res["image"]
        if self.CFG.expand_ch_dim:
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image.astype(np.float32, copy=False))
        else:
            image = img2tensor(image)
        label = get_label(self.df, idx)
        return image, label

class TrainDatasetBowelExtra(Dataset):
    """画像レベルでラベルがあるBowelとExtravasation用のデータセット."""
    def __init__(
        self,
        CFG: Any,
        df: pd.DataFrame,
        preprocess: Union[None, Callable] = None,
        tfms: Union[None, Callable] = None,
    ) -> None:
        self.df = df
        self.CFG = CFG
        self.preprocess = preprocess
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """画像とラベル(mask)の取得.
        Args:
            idx (int): self.dfに対応するデータのインデックス.
        Returns:
            tuple (torch.tensor, torch.tensor): 画像とラベル.
        Note:
            現在読み込む画像の形式は.png, .npy, .npzのみ対応.
        """
        impath = os.path.join(
            self.CFG.image_dir,
            "train_images",
            str(self.df["patient_id"][idx]),
            str(self.df["series_id"][idx]),
            str(self.df["image_id"][idx]) + ".npy",
        )

        if os.path.exists(impath):
            image = load_image(impath)
        else:
            image = np.zeros(self.CFG.image_size)

        if self.preprocess:
            image = self.preprocess(image)

        image = cv2.resize(image, (self.CFG.image_size[1], self.CFG.image_size[0]), interpolation=cv2.INTER_LINEAR)

        if self.tfms:
            res = self.tfms(image=image)
            image = res["image"]

        image = img2tensor(image)
        label = self.get_label(idx)

        return image, label
    
    def get_label(self, idx: int)-> torch.tensor:
        label = [
            self.df["bowel"][idx], 
            self.df["extravasation"][idx]
        ]
        if self.CFG.label_smoothing:
            label = np.clip(label, 0.05, 0.95)
        return torch.tensor(label, dtype=torch.float32)
    
class TestDatasetBowelExtra(Dataset):
    """画像レベルでラベルがあるBowelとExtravasation用のテスト用データセット."""
    def __init__(
        self,
        CFG: Any,
        df: pd.DataFrame,
        preprocess: Union[None, Callable] = None,
        tfms: Union[None, Callable] = None,
    ) -> None:
        self.df = df
        self.CFG = CFG
        self.preprocess = preprocess
        self.tfms = tfms

        assert self.tfms is None

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """画像とラベル(mask)の取得.
        Args:
            idx (int): self.dfに対応するデータのインデックス.
        Returns:
            tuple (torch.tensor, torch.tensor): 画像とラベル.
        Note:
            現在読み込む画像の形式は.png, .npy, .npzのみ対応.
        """
        impath = os.path.join(
            self.CFG.image_dir,
            "train_images",
            str(self.df["patient_id"][idx]),
            str(self.df["series_id"][idx]),
            str(self.df["image_id"][idx]) + ".npy",
        )

        if os.path.exists(impath):
            image = load_image(impath)
        else:
            image = np.zeros(self.CFG.image_size)

        if self.preprocess:
            image = self.preprocess(image)

        image = cv2.resize(image, (self.CFG.image_size[1], self.CFG.image_size[0]), interpolation=cv2.INTER_LINEAR)

        if self.tfms:
            res = self.tfms(image=image)
            image = res["image"]

        image = img2tensor(image)
        label = self.get_label(idx)

        return image, label
    
    def get_label(self, idx: int)-> torch.tensor:
        label = [
            self.df["bowel"][idx], 
            self.df["extravasation"][idx]
        ]
        if self.CFG.label_smoothing:
            label = np.clip(label, 0.05, 0.95)
        return torch.tensor(label, dtype=torch.float32)

def save_df(df: pd.DataFrame, CFG: Any) -> None:
    """DataFrameをcsv形式で保存する.
    Args:
        df (pandas.DataFrame): 保存するDataFrame.
        path (str): 保存先のパス.
    """
    path = os.path.join(CFG.model_save_dir, CFG.exp_name, CFG.exp_name + "_df.csv")
    df.to_csv(path, index=False)


def load_df(CFG: Any) -> pd.DataFrame:
    """csv形式で保存されたDataFrameを読み込む.
    Args:
        path (str): 読み込むファイルのパス.
    Returns:
        pandas.DataFrame: 読み込んだDataFrame.
    """
    path = os.path.join(CFG.model_save_dir, CFG.exp_name, CFG.exp_name + "_df.csv")
    df = pd.read_csv(path)
    return df