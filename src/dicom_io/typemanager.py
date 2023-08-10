"""複雑なdicomタグの型の処理を行う."""
from typing import List

import numpy as np
import pydicom


def pixelspacing_type(pixelspacing: pydicom.multival.MultiValue) -> List[float]:
    """pixelspacingの型を変換する."""
    return [float(x) for x in pixelspacing]


def image_type(image: np.ndarray) -> str:
    """imageの型を取得する."""
    return image.dtype
