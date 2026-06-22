import logging

from ekr.logging_setup import setup_logging


def test_setup_logging_sets_level_and_is_idempotent():
    logger = setup_logging("DEBUG")
    assert logger.name == "ekr"
    assert logger.level == logging.DEBUG
    streams = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(streams) == 1

    setup_logging("INFO")  # 再呼叫一次
    streams2 = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(streams2) == 1          # 不重覆加 handler
    assert logger.level == logging.INFO  # 等級可更新
