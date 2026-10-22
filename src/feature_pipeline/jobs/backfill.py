"""Run once: python -m src.feature_pipeline.jobs.backfill [--start 2024-01-01] [--end YYYY-MM-DD]"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root, so `src` imports in a Hopsworks job

from src.feature_pipeline import backfill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=None, help="default: the last day complete everywhere")
    a = ap.parse_args()
    report = backfill(a.start, a.end)
    print(report)
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())
