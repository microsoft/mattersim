# -*- coding: utf-8 -*-
import os
import sys

from loguru import logger

handlers = {}


def get_logger():
    if not handlers:
        logger.remove()
        handlers["console"] = logger.add(
            sys.stdout,
            colorize=True,
            filter=log_filter,
            enqueue=True,
        )

    return logger


def log_filter(record):
    if record["level"].name != "INFO":
        return True

    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        return True
    else:
        return False
