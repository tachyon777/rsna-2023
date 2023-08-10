"""画像処理機能."""
from typing import List, Tuple

import cv2
import numpy as np


def windowing(
    img: np.ndarray, wl: int = 0, ww: int = 400, mode: str = "uint8"
) -> np.ndarray:
    """windowing process.

    Args:
        img (numpy.ndarray): The input image to filter.
        wl (int): The center of the window.
        ww (int): The width of the window.
        mode (str): The output data type of the filtered image.
            One of {'uint8', 'uint16', 'float32'}.

    Returns:
        numpy.ndarray: The filtered image.

    Raises:
        ValueError: If the mode is not one of {'uint8', 'uint16', 'float32'}.
    """
    floor, ceil = wl - ww // 2, wl + ww // 2
    img = np.clip(img, floor, ceil)
    if mode == "uint8":
        img = (((img - floor) / (ceil - floor)) * 255).astype(np.uint8)
    elif mode == "uint16":
        img = (((img - floor) / (ceil - floor)) * 65535).astype(np.uint16)
    elif mode == "float32":
        img = ((img - floor) / (ceil - floor)).astype(np.float32)
    else:
        raise Exception(f"unexpected mode: {mode}")

    return img


def mri_normalization(img: np.ndarray) -> np.ndarray:
    """MRI画像の正規化.
    Args:
        img (numpy.ndarray): MRI信号の2次元の入力画像.
    Returns:
        numpy.ndarray: 0~1に正規化された画像.
    Note:
        外れ値の除去を含むので、元の信号を復元することはできない.
    """
    img = img.astype(np.float32) / np.percentile(img, 99)
    return img


def image_binarize(img: np.ndarray, meta: dict, threshold: int = -200) -> np.ndarray:
    """入力画像を二値化する.

    Args:
        img (numpy.ndarray): 2次元の入力画像.
        meta (dict): dicomヘッダのメタデータ.
        threshold (int): 画素値の閾値.

    Returns:
        numpy.ndarray: 二値化された画像.

    Note:
        MRIの二値化について:
            MRIの場合、thresholdを参照しない.
            MRIはスライス間で画素値が大幅に異なることがあり、スライスごとに正規化を行う.
            画素値の99%点を最大値として正規化を行う.
            正規化後、0.05を閾値として二値化を行う.
    Todo:
        - MRI画像の二値化処理に対応する。

    """
    if meta["Modality"] not in ("CT", "MR"):
        modal = meta["Modality"]
        raise Exception(f"現在このモダリティには対応していません: {modal}")

    if meta["Modality"] == "MR":
        img_norm = mri_normalization(img)
        img_bin = (img_norm > 0.05).astype(np.uint8)
        return img_bin

    if meta["Modality"] == "CT":
        img_bin = (threshold < img).astype(np.uint8)
        return img_bin


def find_body_contour(img_bin: np.ndarray) -> list:
    """find body surface from binarized image.

    Note:
        cv2.RETR_EXTERNALにより、最外輪郭のみ取得している
    """
    contours = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0]  # 付帯情報の除去
    return contours