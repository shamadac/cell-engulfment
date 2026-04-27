from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = output_dir.name
    log_path = output_dir / f"process_log_{timestamp}.txt"

    logger = logging.getLogger("cell_engulfment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.output_dir = output_dir  # type: ignore[attr-defined]
    logger.log_path = log_path  # type: ignore[attr-defined]
    return logger
