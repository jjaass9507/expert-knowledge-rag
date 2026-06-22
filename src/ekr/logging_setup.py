"""集中設定 ekr 套件的 logging，讓 LLM 請求/回應等訊息真正輸出。

預設只設定 `ekr` 這個 logger（非 root），避免 requests/urllib3 在 DEBUG 灌爆主控台。
等級來源：參數 → 環境變數 EKR_LOG_LEVEL → 預設 INFO。設 EKR_LOG_LEVEL=DEBUG 可看完整 prompt/回應。
"""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def setup_logging(level: str | None = None, logfile: str | None = None) -> logging.Logger:
    """設定並回傳 `ekr` logger；冪等（重複呼叫不會重覆加 handler）。"""
    lvl = (level or os.environ.get("EKR_LOG_LEVEL") or "INFO").upper()
    logger = logging.getLogger("ekr")
    logger.setLevel(lvl)
    logger.propagate = False  # 不往 root 傳，避免重覆輸出

    fmt = logging.Formatter(_FORMAT)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    path = logfile or os.environ.get("EKR_LOG_FILE")
    if path and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
