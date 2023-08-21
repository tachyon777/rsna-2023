import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# 実際のダイス係数がどうなっているか
class DiceCoef(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCoef, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss.
    Note:
        https://github.com/naivelamb/kaggle-cloud-organization/blob/master/losses.py
    """

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        eps = 1e-9
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1).float()
        intersection = torch.sum(m1 * m2, 1)
        union = torch.sum(m1, dim=1) + torch.sum(m2, dim=1)
        score = (2 * intersection + eps) / (union + eps)
        score = (1 - score).mean()
        return score


def multiclass_dice_loss(logits, targets):
    """クラスごとに均等な重み付けを行うことにより、クラス不均衡にロバストにする."""
    loss = 0
    dice = SoftDiceLoss()
    num_classes = targets.size(1)
    for class_nr in range(num_classes):
        loss += dice(logits[:, class_nr, :, :], targets[:, class_nr, :, :])
    return loss / num_classes


class BCEDiceLoss(nn.Module):
    """BCEとDiceの組み合わせ.
    Note:
        BCELossの比重が高い理由は、基本的にはBCELossの方が損失関数としての性質が良いから.
        経験的に学習の崩壊が起きにくい.
    """

    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.dice = multiclass_dice_loss
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # both of input shoud be logits
        bceloss = self.BCE(inputs, targets)
        diceloss = self.dice(inputs, targets)

        return bceloss * 0.75 + diceloss * 0.25


def calculate_conf(pred: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    input should be logits
    treat target.size(1) as n_classes
    return np array [[intersection,union]*class]
    """
    probs = F.sigmoid(pred)
    num = targets.size(0)
    # intersection = torch.sum(m1 * m2, 1)
    num_classes = targets.size(1)
    ret = np.zeros((num_classes, 2), dtype=np.float32)
    for class_nr in range(num_classes):
        m1 = probs[:, class_nr, :, :].view(num, -1)
        m2 = targets[:, class_nr, :, :].view(num, -1).float()
        intersection = torch.sum(m1 * m2)
        union = torch.sum(m1) + torch.sum(m2)
        ret[class_nr][0] = intersection
        ret[class_nr][1] = union

    return ret
