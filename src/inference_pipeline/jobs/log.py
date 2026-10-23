"""The log job: python -m src.inference_pipeline.jobs.log

Every 6 hours, reads the predictor's inference log topic from where the last run stopped and writes the predictions
feature group."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root, so `src` imports in a Hopsworks job

from src.inference_pipeline.log import run_log


def main():
    print(f"{run_log()} prediction rows written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
