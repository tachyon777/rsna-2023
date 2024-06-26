import os
from typing import Any, Callable, Tuple, Union

import cv2
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


def resize(image: np.ndarray, imsize: Tuple[int, int]) -> np.ndarray:
    """画像のリサイズ.
    Args:
        image (numpy.ndarray): 画像.
        imsize (tuple): リサイズ後の画像サイズ.
    Returns:
        numpy.ndarray: リサイズ後の画像.
    Note:
        cv2.INTER_LINEAR: バイリニア補間.
        引数のimsizeは[y, x]であるが、cv2.resizeは[x, y]であることに注意.
    """
    image = cv2.resize(image, (imsize[1], imsize[0]), interpolation=cv2.INTER_LINEAR)
    return image


def img2tensor(img, dtype: np.dtype = np.float32):
    """numpy.ndarrayをtorch.tensorに変換する."""
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class TrainDataset(Dataset):
    """学習用データセット."""

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

    def __getitem__(self, idx: int) -> tuple:
        """画像とラベル(mask)の取得.
        Args:
            idx (int): self.dfに対応するデータのインデックス.
        Returns:
            tuple (torch.tensor, torch.tensor): 画像とラベル(mask).
        Note:
            現在読み込む画像の形式は.png, .npy, .npzのみ対応.
        """
        impath = self.df["image_path"][idx]
        maskpath = self.df["mask_path"][idx]

        image = load_image(impath)
        if maskpath is not None:
            mask = load_image(maskpath)
        else:
            mask = np.zeros((image.shape + (self.CFG.n_class,)))

        if self.preprocess:
            image, mask = self.preprocess(image, mask)

        image = resize(image, self.CFG.image_size)
        mask = resize(mask, self.CFG.image_size)

        if self.tfms:
            res = self.tfms(image=image, mask=mask)
            image, mask = res["image"], res["mask"]

        return img2tensor(image), img2tensor(mask)

class TrainDataset(Dataset):
    """学習用データセット."""

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

    def __getitem__(self, idx: int) -> tuple:
        """画像とラベル(mask)の取得.
        Args:
            idx (int): self.dfに対応するデータのインデックス.
        Returns:
            tuple (torch.tensor, torch.tensor): 画像とラベル(mask).
        Note:
            現在読み込む画像の形式は.png, .npy, .npzのみ対応.
        """
        impath = self.df["image_path"][idx]
        maskpath = self.df["mask_path"][idx]

        image = load_image(impath)
        if maskpath is not None:
            mask = load_image(maskpath)
        else:
            mask = np.zeros((image.shape + (self.CFG.n_class,)))

        if self.preprocess:
            image, mask = self.preprocess(image, mask)

        image = resize(image, self.CFG.image_size)
        mask = resize(mask, self.CFG.image_size)

        if self.tfms:
            res = self.tfms(image=image, mask=mask)
            image, mask = res["image"], res["mask"]

        return img2tensor(image), img2tensor(mask)

class TestDataset(Dataset):
    """テスト用データセット.
    Note:
        - 学習時は、1組の画像とラベルのペアを返していた.
        - 評価時は患者ごとの評価を行うため、患者ごとにインスタンスを建てる.
        - データセットごとにこのクラスの中身を実装し直す必要あり.
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

        assert self.tfms is None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """画像とラベル(mask)の取得.
        Args:
            idx (int): self.dfに対応するデータのインデックス.
        Returns:
            tuple (torch.tensor, torch.tensor): 画像とラベル(mask)のボリュームデータ.
        Note:
            現在読み込む画像の形式は.png, .npy, .npzのみ対応.
        """
        impath = self.df["image_path"][idx]
        maskpath = self.df["mask_path"][idx]

        image = load_image(impath)
        if type(maskpath) is str:
            mask = load_image(maskpath)
        else:
            mask = np.zeros((image.shape + (self.CFG.n_class,)))

        if self.preprocess:
            image, mask = self.preprocess(image, mask)

        image = resize(image, self.CFG.image_size)
        mask = resize(mask, self.CFG.image_size)

        return img2tensor(image), img2tensor(mask)


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
