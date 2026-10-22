"""The hourly job: python -m src.feature_pipeline.jobs.hourly"""

import sys

from src.feature_pipeline import run_latest


def main():
    report = run_latest()
    print(report)
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())
