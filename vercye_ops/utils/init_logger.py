import logging

import colorlog


def get_logger():
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",  # %(module)-15s
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "gray",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger = colorlog.getLogger("logger")

    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

    return logger
