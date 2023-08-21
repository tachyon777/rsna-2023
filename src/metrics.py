"""コンペの評価指標を再現するスクリプト.
特定の臓器に関する学習時の指標としても使うので、ある程度汎用的に書く.
Reference:
    https://www.kaggle.com/code/metric/rsna-trauma-metric/notebook
"""
from typing import Optional, List

import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

sample_weight = {
    "healthy": 1,
    "low": 2,
    "high": 4,
    "bowel": 2,
    "extravasation": 6,
    "any": 6,
}


def normalize_probabilities_to_one(pred: np.ndarray) -> np.ndarray:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    pred_sum = pred.sum(axis=1)
    pred = pred / pred_sum[:, np.newaxis]
    return pred


def logloss(
    pred: np.ndarray,
    label: np.ndarray,
    norm: bool = False,
    grade: Optional[np.ndarray] = None,
) -> np.ndarray:
    """loglossを計算する.
    Args:
        pred (np.ndarray): 予測ラベル. (B,C)
        label (np.ndarray): 正解ラベル. (B,C)
        norm (bool): Trueの場合、各行の和を1に正規化する. (default: False)
        grade (Optional[np.ndarray]): sample_weight (default: None)
    Returns:
        float: logloss.
    """
    if norm:
        pred = normalize_probabilities_to_one(pred)
    # label normが入っている場合に、labelをバイナリ化する.
    label = (label > 0.5).astype(np.float32)
    pred = np.nan_to_num(pred, 0.0)
    result = sklearn.metrics.log_loss(y_true=label, y_pred=pred, sample_weight=grade)
    return result
