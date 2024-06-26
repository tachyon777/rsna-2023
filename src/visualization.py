"""表示・可視化系のソースコード."""
from typing import Union, Optional, Any
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np
import pandas as pd

from src.image_processing import windowing


def view_image(
    img: np.ndarray,
    windowing_process: bool = True,
    wl: int = 0,
    ww: int = 400,
    title: Optional[str] = None,
) -> None:
    """画像の表示を行う."""
    if windowing_process:
        img = windowing(img, wl, ww)
    plt.figure(figsize=(7, 7))
    plt.imshow(img, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def print_injury(df: pd.DataFrame, pid: int) -> None:
    """pid, sidの患者の損傷情報を出力する.
    Args:
        df (pd.DataFrame): train.csvのDataFrame.
        pid (int): 患者ID.
    """
    tmp = df[(df["patient_id"] == pid)].iloc[0]
    info = ""
    if tmp["bowel_injury"]:
        info += "bowel_injury \n"
    if tmp["extravasation_injury"]:
        info += "extravasation_injury \n"
    for organ in ["kidney", "liver", "spleen"]:
        for grade in ["low", "high"]:
            if tmp[f"{organ}_{grade}"]:
                info += f"{organ}_{grade} \n"
    if info == "":
        info = "healty patient"
    else:
        info = "===injury information===\n" + info
    print(info)


def animate(ims: Union[list, np.ndarray]) -> animation.FuncAnimation:
    """画像のアニメーションを作成する.
    Args:
        ims (list or np.ndarray): (Z, H, W)形式の画像の配列.
    """
    skip = 1
    if len(ims) > 200:
        skip = len(ims) // 200
        ims = ims[::skip]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    frame = ax.text(450, 20, "1", size=15, color="white")
    im = ax.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        frame.set_text(str(i * skip + 1))
        return [im]

    return animation.FuncAnimation(
        fig, animate_func, frames=len(ims), interval=1000 // 24
    )


cmaps = {
    0: [246, 173, 60],
    1: [170, 170, 255],
    2: [20, 255, 20],
    3: [255, 30, 100],
    4: [255, 255, 0],
    5: [43, 52, 144],
    6: [60, 175, 48],
    7: [240, 20, 50],
    8: [0, 128, 128],
    9: [255, 255, 50],
    10: [20, 255, 20],
    11: [0, 255, 0],
    12: [128, 200, 250],
    13: [255, 0, 255],
    14: [0, 175, 236],
}


def apply_colormap_to_multilabel_images(CFG: Any, img: np.ndarray) -> np.ndarray:
    """多ラベル画像にカラーマップを適用する.
    Args:
        img: channel last (H, W, C)の多ラベル画像.
    Returns:
        numpy.ndarray: channel last (H, W, 3)のカラーマップ画像.
    """
    cmap = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for c in range(CFG.n_class):
        mp = (img[..., c]).astype(np.uint8).copy()
        cmap_res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cmap_res[:, :, :] = cmaps[c]
        cmap_res = cmap_res * mp[:, :, np.newaxis]
        cmap = cmap | cmap_res.astype(np.uint8)

    return cmap
