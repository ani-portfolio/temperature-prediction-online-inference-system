"""The monitoring job: python -m src.monitoring_pipeline.jobs.daily

Every day after the feature job has written yesterday's actuals and the log job has written yesterday's requests:
scores the logged predictions, reports each shadow model against the champion, and runs the drift, performance, and
served share checks. A flagged check is for the owner to read, not a failure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root, so `src` imports in a Hopsworks job

from src.monitoring_pipeline import run_daily


def main():
    report = run_daily()
    print(f"as of {report.as_of}: {report.scored_rows} predictions scored, {len(report.gate_reports)} gate reports")
    for r in report.gate_reports:
        print(f"  v{r['model_version']} ({r['status']}) vs champion v{r['champion_version']}: {r['days']} days, "
              f"MAE {r['model_mae']} vs {r['champion_mae']}, p {r['p_one_sided']:.3g}")
    print("\n".join(f"FLAGGED: {f}" for f in report.flagged) or "nothing flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
