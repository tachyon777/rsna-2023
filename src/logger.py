"""loggerを取得する."""
import logging
import os
import time
from typing import Any

import logzero


def get_logger(CFG: Any):
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    yyyymmddhhmmss = time.strftime("%Y%m%d%H%M%S")
    filename = f"{CFG.exp_name}_{yyyymmddhhmmss}.log"
    log_path = os.path.join(CFG.model_save_dir, CFG.exp_name, filename)
    logger = logzero.setup_logger(
        name="log0",  # loggerの名前、複数loggerを用意するときに区別できる
        logfile=log_path,  # ログファイルの格納先
        level=20,  # 標準出力のログレベル
        formatter=formatter,  # ログのフォーマット
        maxBytes=65535,  # ログローテーションする際のファイルの最大バイト数
        backupCount=3,  # ログローテーションする際のバックアップ数
        fileLoglevel=20,  # ログファイルのログレベル
        disableStderrLogger=False,  # 標準出力しないかどうか
    )
    return logger