"""カスタムのデータ拡張モジュール."""
import numpy as np

def custom_3d_aug(arr: np.ndarray)->np.ndarray:
    """3次元配列の左右上下フリップと、軸の入れ替えを行う.
    Args:
        arr (np.ndarray): 3次元配列. (Z, H, W)
    Returns:
        np.ndarray: 3次元配列. (Z, H, W)
    """
    if np.random.rand() > 0.5:
        arr = arr[::-1]
    if np.random.rand() > 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() > 0.5:
        arr = arr[:, :, ::-1]
    # 0, 1, 2をランダムな順番に並び替える.
    # arr = arr.transpose(np.random.permutation(3))
    
    return arr
