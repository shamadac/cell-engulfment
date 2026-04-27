# cell-engulfment

Scientific analysis pipeline for quantifying *Haemophilus influenzae* engulfment by *Saccharomyces cerevisiae* from 3D fluorescence microscopy exports.

## Features

- Loads paired `*_hflu.csv` and `*_scer.csv` files exported from ImageJ, or processes raw Nikon `.nd2` files directly with a Python-native backend.
- Segments hflu and scer objects in 3D, measures voxel-scaled morphology, and applies rule-based QC filters.
- Classifies engulfment using exact 3D mask containment, with the original sphere-based heuristic retained for legacy CSV sessions.
- Aggregates per-sample results across biological and technical replicates.
- Computes replicate statistics plus ANOVA or Kruskal-Wallis selection.
- Generates figures and size-filter diagnostics.
- Writes QC overlays, performance reports, and resumable per-sample caches for ND2 sessions.
- Optionally runs an ImageJ macro on raw ND2 sessions as a legacy fallback.

## Project Layout

- `config.example.yaml`: Template configuration for local runs.
- `src/`: Python pipeline modules.
- `macros/image_processing.ijm`: ImageJ macro for ND2 preprocessing.
- `tests/`: Automated tests and fixtures.

## Installation

```bash
python -m pip install -r requirements.txt
```

The ND2 backend depends on:

- `nd2` for Nikon raw-file access
- `scikit-image` and `scipy.ndimage` for segmentation, labeling, morphology, and measurements

## Configuration

Copy the example config before running locally:

```bash
cp config.example.yaml config.yaml
```

On Windows PowerShell:

```powershell
Copy-Item config.example.yaml config.yaml
```

`config.yaml` controls:

- hflu and scer size-filter thresholds
- one or more microscopy sessions
- each session's `csv_dir`
- optional per-session `nd2_dir`
- output base directory
- optional `imagej_executable`
- pipeline backend, worker count, cache directory, QC overlay output, label-stack output, and local staging mode
- per-channel segmentation parameters for `scer` and `hflu`
- engulfment method and assignment settings
- figure DPI, format, and violin-plot toggle

`config.yaml` is intentionally ignored by git because it usually contains local data paths. Commit changes to `config.example.yaml` when shared defaults need to change.

## Usage

Run the pipeline:

```bash
python src/main.py
```

Run with a different config:

```bash
python src/main.py --config tests/fixtures/test_config.yaml
```

Run the tests:

```bash
python -m pytest -q
```

## Output

Each run creates a timestamped directory under `output/` containing:

- `process_log_<timestamp>.txt`
- `summary.csv`
- `qc_summary.csv`
- `performance.csv`
- `statistical_report.csv`
- `per_sample/results_<sample>.csv`
- `per_sample/assignment_<sample>.csv` when assignment export is enabled
- `figures/*.png`
- `cell_sizes/*.png`
- `qc/*_overlay.png`
- `qc/*_scer_max.png`
- `qc/*_hflu_max.png`

## Notes

- Sample prefixes must match `^[A-Z][1-9][0-9]*$`.
- Raw microscopy files, derived exports, caches, and run outputs are intentionally excluded from git.
- Shapiro-Wilk runs only for biological replicates with at least 3 technical replicates.
- If any replicate has fewer than 3 observations, or any normality test fails, the pipeline uses Kruskal-Wallis for the group comparison.
