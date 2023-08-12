"""深層学習モデルの読み込み."""
import os
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn


class unetv2(nn.Module):
    def __init__(self, CFG: Any):
        super(unetv2, self).__init__()
        encoder_weights = "imagenet" if CFG.pretrain is True else CFG.pretrain
        self.cnn_model = smp.Unet(
            CFG.backbone,
            encoder_weights=encoder_weights,
            classes=CFG.n_class,
            activation=None,
            in_channels=CFG.n_ch,
        )

    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs


class unetv2_amp(unetv2):
    def __init__(self, *args):
        super(unetv2_amp, self).__init__(*args)

    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(unetv2_amp, self).forward(*args)


def lock_model_encoder_weight(model: Any, mode: str) -> Any:
    """事前学習済みモデルのエンコーダー部分の重みを固定する.
    最終層と、セグメンテーションモデルのデコーダのみ学習可能とする.
    Args:
        model (Any): モデル
        mode (str): モード. "lock" or "unlock"
    Note:
        smp_pytorchの、cnnモデルでのみhead_nameのパラメータが使用可能.
        smp以外のフレームワークやViT系を使ったりする場合は適宜書き換える必要がある.
    """
    head_name = [
        "cnn_model.encoder._conv_head.weight",
        "cnn_model.encoder._bn1.weight",
        "cnn_model.encoder._bn1.bias",
    ]

    if mode == "lock":
        for hname, param in model.named_parameters():
            if hname in head_name or "decoder" in hname or "segmentation_head" in hname:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif mode == "unlock":
        for hname, param in model.named_parameters():
            param.requires_grad = True
    else:
        raise Exception("mode is 'lock' or 'unlock'")

    return model


def load_models(CFG: Any, mode: str = "final") -> list:
    """Configの内容から、学習済みの全モデルを読み込む."""
    assert mode in ["final", "best"]
    models = []
    for fold in range(CFG.train_folds):
        model = unetv2(CFG)
        params_path = os.path.join(
            CFG.model_save_dir, CFG.exp_name, f"{CFG.exp_name}_f{fold}_{mode}.pth"
        )
        params = torch.load(params_path, map_location=torch.device("cpu"))
        model.load_state_dict(params)
        model.float()
        model.eval()
        model.to(CFG.device)
        models.append(model)
    return models