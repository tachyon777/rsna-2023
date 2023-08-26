import os
import time
from typing import Any

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from src.metrics import logloss


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
    y_pred_list = []
    y_list = []
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
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.detach().to(torch.float32).cpu()
            y = y.to(torch.float32).cpu()
            epoch_loss += loss_np
            y_pred_list.append(y_pred)
            y_list.append(y)

            if CFG.progress_bar:
                bar.set_description("Validation loss: %.5f" % (loss_np))
    y_pred_list = torch.cat(y_pred_list, dim=0)
    y_list = torch.cat(y_list, dim=0)
    val_metric = logloss(y_pred_list, y_list)
    metric_mean = val_metric.mean()

    return epoch_loss / len(iterator), metric_mean


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
    best_valid_score = 10 ** 9

    train_losses = []
    valid_losses = []
    valid_metrics = []

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
        valid_loss, valid_metric = validate(CFG, model, valid_iterator, loss_criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_metrics.append(valid_metric)

        # best scoreを更新していたならモデルを保存
        if valid_metric < best_valid_score:
            best_valid_score = valid_metric
            if CFG.num_gpus == 1:
                torch.save(
                    model.state_dict(),
                    os.path.join(CFG.model_save_dir, CFG.exp_name, f"{name}_best.pth"),
                )
            else:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(CFG.model_save_dir, CFG.exp_name, f"{name}_best.pth"),
                )

        end_time = time.time()

        epoch_mins, epoch_secs = (
            (end_time - start_time) // 60,
            round((end_time - start_time) % 60),
        )
        logger.info(
            f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins:.0f}m {epoch_secs}s"
        )
        logger.info(f'lr:{optimizer.param_groups[0]["lr"]:.7f}')
        logger.info(f"Train Loss: {train_loss:.3f}")
        logger.info(
            f"Val. Loss: {valid_loss:.3f} | Val. Logloss Score : {valid_metric:.3f},"
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

    return train_losses, valid_losses, valid_metrics


def evaluate(CFG: Any, models: list, iterator: Any) -> tuple:
    """推論用関数.
    評価指標を算出せずに、正解label、推論をそのまま返す.
    """
    y_r = []
    p_r = []
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(CFG.device).to(torch.float32)
            y = y.to(CFG.device).to(torch.float32)
            yres, pres = None, None
            for model in models:
                model.eval()
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred)
                if pres is None:
                    pres = y_pred.detach().cpu().numpy()
                    yres = y.detach().cpu().numpy()
                else:
                    pres += y_pred.detach().cpu().numpy()
            pres /= len(models)

            for i in range(yres.shape[0]):
                y_r.append(yres[i])
                p_r.append(pres[i])
    return {"label": np.array(y_r), "pred": np.array(p_r)}


def inference(CFG: Any, models: list, iterator: Any) -> np.ndarray:
    """正解ラベルが与えられない推論用の関数.
    iteratorが1要素(x)のみ渡すことに注意.
    """
    p_r = []
    with torch.no_grad():
        for x in iterator:
            x = x.to(CFG.device).to(torch.float32)
            pres = None
            for model in models:
                model.eval()
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred)
                if pres is None:
                    pres = y_pred.detach().cpu().numpy()
                else:
                    pres += y_pred.detach().cpu().numpy()
            pres /= len(models)

            for i in range(pres.shape[0]):
                p_r.append(pres[i])
                
    return np.array(p_r)
