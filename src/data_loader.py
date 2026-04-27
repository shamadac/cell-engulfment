from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "XM",
    "YM",
    "ZM",
    "BX",
    "BY",
    "BZ",
    "B-width",
    "B-height",
    "B-depth",
    "Volume (micron^3)",
)

SAMPLE_PREFIX_RE = re.compile(r"^[A-Z][1-9][0-9]*$")


@dataclass(frozen=True)
class Sample:
    prefix: str
    session_label: str
    biological_replicate: str
    technical_replicate: int
    hflu_path: Path
    scer_path: Path


def _extract_prefix(path: Path, suffix: str) -> str | None:
    if not path.name.endswith(suffix):
        return None
    return path.name[: -len(suffix)]


def _is_valid_prefix(prefix: str) -> bool:
    return bool(SAMPLE_PREFIX_RE.match(prefix))


def discover_samples(session_dir: Path, session_label: str, logger) -> list[Sample]:
    hflu_by_prefix: dict[str, Path] = {}
    scer_by_prefix: dict[str, Path] = {}

    for csv_path in sorted(session_dir.glob("*.csv")):
        prefix: str | None = None
        if csv_path.name.endswith("_hflu.csv"):
            prefix = _extract_prefix(csv_path, "_hflu.csv")
            if prefix and _is_valid_prefix(prefix):
                hflu_by_prefix[prefix] = csv_path
            elif prefix:
                logger.warning("Skipping invalid hflu filename prefix '%s' in %s", prefix, csv_path)
        elif csv_path.name.endswith("_scer.csv"):
            prefix = _extract_prefix(csv_path, "_scer.csv")
            if prefix and _is_valid_prefix(prefix):
                scer_by_prefix[prefix] = csv_path
            elif prefix:
                logger.warning("Skipping invalid scer filename prefix '%s' in %s", prefix, csv_path)

    samples: list[Sample] = []
    all_prefixes = sorted(set(hflu_by_prefix) | set(scer_by_prefix), key=lambda value: (value[0], int(value[1:])))
    for prefix in all_prefixes:
        hflu_path = hflu_by_prefix.get(prefix)
        scer_path = scer_by_prefix.get(prefix)
        if hflu_path is None or scer_path is None:
            missing = "_hflu.csv" if hflu_path is None else "_scer.csv"
            logger.warning("Skipping unpaired sample '%s' in %s; missing %s", prefix, session_dir, missing)
            continue

        samples.append(
            Sample(
                prefix=prefix,
                session_label=session_label,
                biological_replicate=prefix[0],
                technical_replicate=int(prefix[1:]),
                hflu_path=hflu_path,
                scer_path=scer_path,
            )
        )

    return samples


def load_sample_csvs(sample: Sample) -> tuple[pd.DataFrame, pd.DataFrame]:
    hflu_df = pd.read_csv(sample.hflu_path)
    scer_df = pd.read_csv(sample.scer_path)

    missing_hflu = sorted(set(REQUIRED_COLUMNS) - set(hflu_df.columns))
    missing_scer = sorted(set(REQUIRED_COLUMNS) - set(scer_df.columns))

    if missing_hflu:
        raise ValueError(
            f"Sample {sample.prefix} hflu CSV is missing required columns: {', '.join(missing_hflu)}"
        )
    if missing_scer:
        raise ValueError(
            f"Sample {sample.prefix} scer CSV is missing required columns: {', '.join(missing_scer)}"
        )

    return hflu_df, scer_df
