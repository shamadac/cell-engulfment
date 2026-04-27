"""Command-line entry point for the cell-engulfment pipeline."""

from __future__ import annotations

import argparse
import sys

from config_loader import ConfigError, load_config
from pipeline import run_pipeline


def main() -> int:
    """Parse CLI arguments, load configuration, and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the cell-engulfment analysis pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to the pipeline config file.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_pipeline(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
