import logging
from pathlib import Path


def setup_logger():
    """
    Configure the VisionX application logger.

    Returns:
        Configured logger instance.
    """

    log_directory = Path("outputs")
    log_directory.mkdir(exist_ok=True)

    log_file = log_directory / "visionx.log"

    logger = logging.getLogger("VisionX")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger