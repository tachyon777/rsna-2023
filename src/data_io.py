"""データ入出力関連."""
import os
from typing import Any, List, Optional

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


def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """特殊なデータ格納方式の場合にビットシフトを適用する.
    Reference:
        https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
    return pixel_array


def load_image_from_dicom(dicom: pydicom.dataset.FileDataset, meta: dict) -> np.ndarray:
    """load image from dicom path."""
    image = standardize_pixel_array(dicom)
    image = image.astype(np.int16)
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


def load_dicom_series(dir_: str, max_slices: Optional[int]) -> Any:
    """load dicom series from directory.
    Args:
        dir_ (str): dicomファイルが格納されているディレクトリ.
        max_slices (int): 画像の最大枚数. Noneならば全ての画像を読み込む.

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
        - ファイル名を参照してスライス数を決定するため、RSNA2023データにのみ対応.
        - ディレクトリ内にはdicomファイルしか存在せず、スライス番号がファイル名となっている必要がある.
    """

    path_list = [[int(path.split(".")[0]), path] for path in os.listdir(dir_)]
    path_list.sort()
    if max_slices is not None:
        step = (len(path_list) + max_slices - 1) // max_slices
        path_list = path_list[::step]

    tmp_list = []
    for idx, path in path_list:
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

        tmp_list.append((idx, path, dicom, meta, image))

    path_list = []
    meta_list = []
    image_arr = []
    for _, path, dicom, meta, image in tmp_list:
        path_list.append(path)
        meta_list.append(meta)
        if image is not None:
            image_arr.append(image)

    return np.array(image_arr), path_list, meta_list
