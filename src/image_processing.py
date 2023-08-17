"""画像処理機能."""
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d


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

def apply_preprocess(image: np.ndarray, resize: Optional[tuple]=None)-> Tuple[np.ndarray, np.ndarray]:
    """データ前処理. カスタマイズして使用.
    Args:
        image (numpy.ndarray): HU値のCT画像. (z, h, w)
        resize (tuple): リサイズ後の画像サイズ. Noneならばリサイズしない. (h, w)
    Returns:
        image (numpy.ndarray): windowing及び0~1に正規化. (z, h, w)
    """
    # 0~1に正規化
    image = windowing(image, wl=0, ww=400, mode="float32")
    if resize:
        new_arr = []
        for i in range(image.shape[0]):
            new_arr.append(cv2.resize(image[i], (resize[1], resize[0])))
        return np.stack(new_arr)
    
    return image

def crop_organ(image: np.ndarray, mask: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """画像の3D配列に対して、臓器領域を切り抜き、臓器に外接するボリュームを返す."""
    # 臓器が存在する部分のインデックスを取得
    z_indices, h_indices, w_indices = np.where(mask != 0)

    """# 各軸に沿って最小と最大のインデックスを見つける
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    h_min, h_max = np.min(h_indices), np.max(h_indices)
    w_min, w_max = np.min(w_indices), np.max(w_indices)"""

    # 各軸に沿って、p%のボクセルが含まれる範囲を見つける
    p = 98
    z_min, z_max = np.percentile(z_indices, 100-p), np.percentile(z_indices, p)
    h_min, h_max = np.percentile(h_indices, 100-p), np.percentile(h_indices, p)
    w_min, w_max = np.percentile(w_indices, 100-p), np.percentile(w_indices, p)
    z_min, z_max = int(z_min), int(z_max)
    h_min, h_max = int(h_min), int(h_max)
    w_min, w_max = int(w_min), int(w_max)

    # この範囲でセグメンテーションデータを切り抜く
    margin = 10
    z_min, z_max = max(0, z_min - 5), min(image.shape[0], z_max + 5)
    h_min, h_max = max(0, h_min - margin), min(image.shape[1], h_max + margin)
    w_min, w_max = max(0, w_min - margin), min(image.shape[2], w_max + margin)
    cropped_image = image[z_min:z_max+1, h_min:h_max+1, w_min:w_max+1]
    cropped_mask = mask[z_min:z_max+1, h_min:h_max+1, w_min:w_max+1]

    # crop segmentation
    # cropped_image = cropped_image * cropped_mask + (1 - cropped_mask) * -1000

    return cropped_image, cropped_mask

def kidney_split(image: np.ndarray, mask: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """腎臓について、一度crop_organに入力したものを再度この関数に入力することで左右の腎臓に切り出す.
    Args:
        image (numpy.ndarray): image.
        mask (numpy.ndarray): (Z, H, W)のマスク画像.
    Returns:
        left_image, right_image
    Note:
        本関数中のleft/rightは画像上のleft_rightを表す.
    """
    w_half = image.shape[2] // 2
    left_image = image[:, :, :w_half]
    left_mask = mask[:, :, :w_half]
    right_image = image[:, :, w_half:]
    right_mask = mask[:, :, w_half:]
    left_image, _ = crop_organ(left_image, left_mask)
    right_image, _ = crop_organ(right_image, right_mask)
    return left_image, right_image

def resize_volume(mask: np.ndarray, hw_shape: tuple)-> np.ndarray:
    """h, wが512ではない場合にmaskをimageに合うようにリサイズする."""
    new_arr = []
    for i in range(mask.shape[0]):
        new_arr.append(cv2.resize(mask[i], hw_shape[::-1]))
    return np.stack(new_arr)

def morpho_pytorch(CFG: Any, masks: np.ndarray)-> np.ndarray:
    """3次元のモルフォロジー処理をpytorchで行う.
    Note:
        - cpuで演算を行うと非常に時間がかかるので注意.
    """
    for c in range(masks.shape[-1]):
        with torch.no_grad():
            arr = torch.tensor(masks[...,c][np.newaxis]).to(CFG.device).to(torch.float32)
            #dialation
            arr = torch.nn.MaxPool3d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)(arr)
            #erosion
            arr = -torch.nn.MaxPool3d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)(-arr)
        arr = arr.squeeze(0).cpu().numpy().astype(np.uint8)
        masks[...,c] = arr
    return masks

# 各臓器に対して、一定閾値以下のボクセルの集合を切り捨てる
area_th = {
    "liver":50,
    "spleen":20,
    "kidney":20,
    "bowel":30,
}

def area_0fill(masks: np.ndarray, organ_index_dict: dict)-> np.ndarray:
    """各臓器に対して、一定閾値以下のボクセルの集合を切り捨てる."""
    for idx,mask in enumerate(masks):
        for c,th in area_th.items():
            c_idx = organ_index_dict[c]
            
            contours = cv2.findContours(mask[...,c_idx],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] #0番目は元画像,2番目は階層構造。新しいopencvだと1番目のみがコンツーリング情報っぽい
            for con in contours:
                area = cv2.contourArea(con)
                if area <= area_th[c]:
                    fill_mask = mask[...,c_idx].copy()
                    fill_mask = cv2.drawContours(fill_mask, [con], 0, 0, -1)
                    masks[idx,:,:,c_idx] =  fill_mask
    return masks

def apply_postprocess(CFG: Any, mask: np.ndarray)-> np.ndarray:
    """セグメンテーション後の臓器マスクの後処理.
    Args:
        mask (numpy.ndarray): (Z, H, W, C)のマスク画像.
    """
    mask = morpho_pytorch(CFG, mask)
    # mask = area_0fill(mask)
    return mask

def resize_3d(image: np.ndarray, imsize: Tuple[int, int, int]) -> np.ndarray:
    """ボリュームデータのリサイズ.
    Args:
        image (numpy.ndarray): volume(z, h, w).
        imsize (tuple): リサイズ後の画像サイズ(z, h, w).
    Returns:
        numpy.ndarray: resized volume.
    """
    image = cv2.resize(image, (imsize[1], imsize[0]), interpolation=cv2.INTER_LINEAR)
    image = resize_1d(image, imsize[2], axis=2)
    return image

def resize_1d(image: np.ndarray, imsize: int, axis: int = 2) -> np.ndarray:
    """3次元配列のうち、axisに指定した1次元をリサイズする."""
    # 指定された軸を最後に移動
    image_moved = np.moveaxis(image, axis, -1)

    # もし指定したaxisのshapeが1ならば、配列をコピーして拡張.
    # sample submissionでこういう極端な例がある.
    if image_moved.shape[-1] == 1:
        result_moved = np.tile(image_moved, imsize)
    
    else:
        x_old = np.linspace(0, 1, image_moved.shape[-1])
        x_new = np.linspace(0, 1, imsize)
        interpolator = interp1d(x_old, image_moved, axis=-1)

        result_moved = interpolator(x_new)
    # 元の軸の順序に戻す
    result = np.moveaxis(result_moved, -1, axis)
    
    return result

def kidney_specific(CFG, l: np.ndarray, r: np.ndarray) -> np.ndarray:
    """ボリュームサイズの異なる左右の腎臓をW方向にconcatして返す.
    Args:
        l (np.ndarray): 右腎臓のボリューム(Z, H, W).
        r (np.ndarray): 左腎臓のボリューム(Z, H, W). 
    Note:
        - 本関数のl, rは画像上の左右を表す.
        - source: src.classification.dataset.TrainDatasetSolidOrgans.kidney_specific
    """
    # z
    l = resize_1d(l, CFG.image_size[0] ,axis=0)
    r = resize_1d(r, CFG.image_size[0] ,axis=0)
    # h
    l = resize_1d(l, CFG.image_size[1] ,axis=1)
    r = resize_1d(r, CFG.image_size[1] ,axis=1)
    # w
    image = np.concatenate([l, r], axis=2)
    image = resize_1d(image, CFG.image_size[2] ,axis=2)
    return image