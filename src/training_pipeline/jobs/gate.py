"""The gate job: python -m src.training_pipeline.jobs.gate --version N

Started by the train job with the version it registered; run it by hand to judge a candidate again. A rejected
candidate is an answer, not a failure, so the job only fails on an error."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root, so `src` imports in a Hopsworks job

from src.training_pipeline import run_gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", type=int, required=True, help="the model version to judge")
    report = run_gate(parser.parse_args(argv).version)
    print(report)
    if report.next_step:
        print(report.next_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
