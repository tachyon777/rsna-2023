import os
import time
from typing import Any

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from src.loss import calculate_conf


def train(CFG, model, iterator, optimizer, criterion, scaler) -> float:
    """学習を行う.
    fit_trainerから1回呼び出されると1epoch分の学習が行われる.
    Args:
        CFG: config.
        model: モデル.
        iterator: 学習データのイテレータ.
        optimizer: 最適化手法.
        criterion: 損失関数.
        device: デバイス.
        scaler: amp用. デフォルトはNone.
    Returns:
        float: 1epoch分の学習のlossの平均.
    """
    epoch_loss = 0
    model.train()

    # プログレスバーを表示するか否か
    bar = tqdm(iterator) if CFG.progress_bar else iterator

    for x, y in bar:
        x = x.to(CFG.device).to(torch.float32)
        y = y.to(CFG.device).to(torch.float32)
        optimizer.zero_grad()
        if CFG.amp:
            with amp.autocast():
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        epoch_loss += loss_np

        if CFG.progress_bar:
            bar.set_description("Training loss: %.5f" % (loss_np))

    return epoch_loss / len(iterator)


def validate(CFG, model, iterator, criterion) -> tuple:
    epoch_loss = 0
    val_dice_each_cls = np.zeros((CFG.n_class, 2), dtype=np.float32)
    model.eval()

    bar = tqdm(iterator) if CFG.progress_bar else iterator

    with torch.no_grad():
        for x, y in bar:
            x = x.to(CFG.device).to(torch.float32)
            y = y.to(CFG.device).to(torch.float32)

            if CFG.amp:
                with amp.autocast():
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
            else:
                y_pred = model(x)
                loss = criterion(y_pred, y)
            loss_np = loss.detach().cpu().numpy()
            y_pred = y_pred.detach().to(torch.float32).cpu()
            y = y.to(torch.float32).cpu()
            epoch_loss += loss_np
            val_dice_each_cls += calculate_conf(y_pred, y)

            if CFG.progress_bar:
                bar.set_description("Validation loss: %.5f" % (loss_np))

    dice_cls = np.zeros((CFG.n_class,), dtype=np.float32)
    dice_mean = 0
    for c in range(val_dice_each_cls.shape[0]):
        dcs = (2 * val_dice_each_cls[c][0]) / val_dice_each_cls[c][1]
        dice_cls[c] = dcs
        dice_mean += dcs

    dice_mean /= CFG.n_class

    return epoch_loss / len(iterator), dice_mean, dice_cls


def fit_model(
    CFG,
    model,
    name,
    train_iterator,
    train_noaug_iterator,
    valid_iterator,
    optimizer,
    loss_criterion,
    freeze,
    scaler,
    logger,
    scheduler,
    organ_index_dict_inv,
) -> tuple:
    """Fits a dataset to model"""
    best_valid_score = -1

    train_losses = []
    valid_losses = []
    valid_dices = []
    valid_dice_each_clsses = []

    for epoch in range(CFG.freeze_epochs if freeze else CFG.n_epoch + CFG.noaug_epochs):
        # 最後のn世代、augmentationなしで学習
        if (freeze is False) and (epoch == CFG.n_epoch):
            logger.info("-_-_-_-_-_-_-_-_-")
            logger.info("No augment mode")
            logger.info("-_-_-_-_-_-_-_-_-")
            train_iterator = train_noaug_iterator
        if scheduler:
            scheduler.step(epoch)

        start_time = time.time()

        train_loss = train(
            CFG, model, train_iterator, optimizer, loss_criterion, scaler
        )
        valid_loss, valid_dice, valid_dice_each_cls = validate(
            CFG, model, valid_iterator, loss_criterion
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_dices.append(valid_dice)
        valid_dice_each_clsses.append(valid_dice_each_cls)

        # best scoreを更新していたならモデルを保存
        if valid_dice > best_valid_score:
            best_valid_score = valid_dice
            if CFG.num_gpus == 1:
                torch.save(
                    model.state_dict(),
                    os.path.join(CFG.model_save_dir, CFG.exp_name, f"{name}_best.pt"),
                )
            else:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(CFG.model_save_dir, CFG.exp_name, f"{name}_best.pt"),
                )

        end_time = time.time()

        epoch_mins, epoch_secs = (end_time - start_time) // 60, round(
            (end_time - start_time) % 60
        )
        logger.info(
            f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins:.0f}m {epoch_secs}s"
        )
        logger.info(f'lr:{optimizer.param_groups[0]["lr"]:.7f}')
        logger.info(f"Train Loss: {train_loss:.3f}")
        logger.info(
            f"Val. Loss: {valid_loss:.3f} | Val. Dice Score : {valid_dice:.3f},"
        )
        logger.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        for i in range(CFG.n_class):
            logger.info(
                f"{organ_index_dict_inv[i]} Dice Score : {valid_dice_each_cls[i]:.3f}"
            )

    if not freeze:
        if CFG.num_gpus == 1:
            torch.save(
                model.state_dict(),
                os.path.join(CFG.model_save_dir, CFG.exp_name, f"{name}_final.pth"),
            )
        else:
            torch.save(
                model.module.state_dict(),
                os.path.join(CFG.model_save_dir, CFG.exp_name, f"{name}_final.pth"),
            )

    return train_losses, valid_losses, valid_dices, valid_dice_each_clsses


def evaluate(CFG: Any, models: list, iterator) -> tuple:
    """推論用関数.
    評価指標を算出せずに、元画像、正解画像、推論結果(pred_proba, float32)をそのまま返す.
    Note:
        - 入力はchannel firstだが、returnはchannel lastにしている.
    """
    x_r = []
    y_r = []
    p_r = []
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(CFG.device).to(torch.float32)
            y = y.to(CFG.device).to(torch.float32)
            xres, yres, pres = None, None, None
            for model in models:
                model.eval()
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred)
                if pres is None:
                    pres = y_pred.detach().cpu().numpy()
                    xres = x.detach().cpu().numpy()
                    yres = y.detach().cpu().numpy()
                else:
                    pres += y_pred.detach().cpu().numpy()
            pres /= len(models)

            for i in range(xres.shape[0]):
                x_r.append(xres[i].transpose(1, 2, 0))
                y_r.append(yres[i].transpose(1, 2, 0))
                p_r.append(pres[i].transpose(1, 2, 0))
    return {"image": np.array(x_r), "label": np.array(y_r), "pred": np.array(p_r)}