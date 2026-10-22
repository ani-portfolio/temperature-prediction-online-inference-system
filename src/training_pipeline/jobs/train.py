"""The train job: python -m src.training_pipeline.jobs.train"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root, so `src` imports in a Hopsworks job

from src.training_pipeline import run_training


def main():
    print(run_training())
    return 0


if __name__ == "__main__":
    sys.exit(main())
