import logging
import os
import sys


def create_logger(logger_name):
    default_logger = os.getenv("ML_LOGGING", "1")
    if default_logger.lower() in ["true", "1"]:
        default_logger = True
    else:
        default_logger = False

    log_level = logging.DEBUG if os.getenv("ML_DEBUG", False) else logging.INFO

    logger = logging.getLogger(logger_name)

    fmt = logging.Formatter(fmt="[%(asctime)s] - %(message)s", datefmt="%H:%M:%S")

    handler = (
        logging.StreamHandler(sys.stdout) if default_logger else logging.NullHandler()
    )

    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger
