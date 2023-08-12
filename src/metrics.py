from collections import defaultdict

import numpy as np
from monai.metrics import compute_hausdorff_distance


def get_hausdorff_3d(
    label: np.ndarray, pred: np.ndarray, percentile: int = 95
) -> np.ndarray:
    """Hausdorff距離を計算する.
    Args:
        label (np.ndarray): 正解ラベル. (H,W,D)
        pred (np.ndarray): 予測ラベル. (H,W,D)
    Returns:
        np.ndarray: 各クラスごとのHausdorff距離. (n_class,)
    """
    assert label.shape == pred.shape
    HD = []
    for c in range(label.shape[-1]):
        A = label[..., c]
        B = pred[..., c]
        HD_ = compute_hausdorff_distance(
            A[np.newaxis, np.newaxis],
            B[np.newaxis, np.newaxis],
            include_background=True,
            percentile=percentile,
        )
        HD.append(HD_[0][0].numpy())
    return np.array(HD)


def calc_cfm(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """混同行列を計算する.
    Args:
        label (np.ndarray): 正解ラベル. (H,W,D,C)
        pred (np.ndarray): 予測ラベル. (H,W,D,C)
    Returns:
        np.ndarray: 各クラスごとの混同行列. (n_class,4)
    Note:
        - 混同行列の各要素は、[TP,TN,FP,FN]の順番.
        - Precision, Recall, Dice係数などの算出に用いる.
    """
    assert label.shape == pred.shape
    cfm = []
    for c in range(label.shape[-1]):
        A = label[..., c]
        B = pred[..., c]
        TP = ((A == 1) & (B == 1)).sum()
        TN = ((A == 0) & (B == 0)).sum()
        FP = ((A == 0) & (B == 1)).sum()
        FN = ((A == 1) & (B == 0)).sum()
        cfm.append([TP, TN, FP, FN])

    return np.array(cfm)


def calc_cfm_metrics(label: np.ndarray, pred: np.ndarray) -> dict:
    """混同行列を算出し、Precision, Recall, Dice係数を計算する.
    Args:
        label (np.ndarray): 正解ラベル. (H,W,D,C)
        pred (np.ndarray): 予測ラベル. (H,W,D,C)
    Returns:
        dict: 各クラスごとのPrecision, Recall, Dice係数. (n_class,3)
    """
    cfm = calc_cfm(label, pred)
    metrics = defaultdict(list)
    for c in range(label.shape[-1]):
        TP, TN, FP, FN = cfm[c]
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["dice"].append(dice)
    return metrics