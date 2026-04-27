"""Wrapper for the optional legacy ImageJ/Fiji preprocessing macro."""

from __future__ import annotations

import subprocess
from pathlib import Path


MACRO_ARG_SEPARATOR = "|"


def run_imagej_macro(
    imagej_path: str | Path,
    nd2_dir: Path,
    csv_output_dir: Path,
    macro_path: Path,
    logger,
) -> bool:
    """Run the ImageJ macro and report success without raising subprocess errors."""
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    (csv_output_dir / "processed_images").mkdir(parents=True, exist_ok=True)

    command = [
        str(imagej_path),
        "-batch",
        str(macro_path),
        # Windows microscopy folders can contain commas, so use a forbidden path character.
        f"{nd2_dir}{MACRO_ARG_SEPARATOR}{csv_output_dir}",
    ]

    logger.info("Running ImageJ macro: %s", " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        logger.error("ImageJ macro failed for %s with exit code %s", nd2_dir, completed.returncode)
        if completed.stdout:
            logger.error("ImageJ stdout:\n%s", completed.stdout)
        if completed.stderr:
            logger.error("ImageJ stderr:\n%s", completed.stderr)
        return False

    if completed.stdout:
        logger.info("ImageJ stdout:\n%s", completed.stdout)
    return True
