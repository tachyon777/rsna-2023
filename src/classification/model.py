"""深層学習モデルの読み込み."""
import os
from typing import Any

import torch
from torch import nn
import timm

from models.efficientnet_pytorch_3d import EfficientNet3D


def lock_model_encoder_weight(model: Any, mode: str) -> Any:
    """事前学習済みモデルのエンコーダー部分の重みを固定する.
    最終層のみ学習可能とする.
    Args:
        model (Any): モデル
        mode (str): モード. "lock" or "unlock"
    Note:
        smp_pytorchの、cnnモデルでのみhead_nameのパラメータが使用可能.
        timm以外のフレームワークやViT系を使ったりする場合は適宜書き換える必要がある.
    """

    enet3d_b1_head = [
        "_conv_head.weight",
        "_bn1.weight",
        "_bn1.bias",
        "_fc.weight",
        "_fc.bias",
    ]

    enet3d_b4_head = [
        "module._conv_head.weight",
        "module._bn1.weight",
        "module._bn1.bias",
        "module._fc.weight",
        "module._fc.bias",
    ]
    timm_head = [
        "module.conv_head.weight",
        "module.bn2.weight",
        "module.bn2.bias",
        "module.classifier.weight",
        "module.classifier.bias",
    ]
    heads = enet3d_b1_head + enet3d_b4_head + timm_head
    flag = 0
    if mode == "lock":
        for hname, param in model.named_parameters():
            if hname in heads:
                flag = 1
                param.requires_grad = True
            else:
                param.requires_grad = False
        if flag == 0:
            raise Exception("head_name is not found")
    elif mode == "unlock":
        for hname, param in model.named_parameters():
            param.requires_grad = True
    else:
        raise Exception("mode is 'lock' or 'unlock'")

    return model


def load_models(
    CFG: Any, mode: str = "final", framework: str = "EfficientNet3D"
) -> list:
    """Configの内容から、学習済みの全モデルを読み込む."""
    assert framework in ["EfficientNet3D", "timm"]
    models = []
    for fold in range(CFG.train_folds):
        if framework == "EfficientNet3D":
            model = EfficientNet3D.from_name(
                CFG.backbone,
                override_params={"num_classes": CFG.n_class},
                in_channels=CFG.n_ch,
            )
        elif framework == "timm":
            model = timm.create_model(
                CFG.backbone,
                pretrained=False,
                num_classes=CFG.n_class,
                in_chans=CFG.n_ch,
            )
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
