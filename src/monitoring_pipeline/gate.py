"""The live gate: each shadow model against the champion on live days, the evidence for a promotion or a rollback.

Both models are scored on the same city-days, after both training cutoffs, from the first such day on (no fixed
window). The one sided Diebold-Mariano test from src.compare asks whether the shadow's MAE is lower: for a challenger
a significant result supports a promotion, and for a retiring former champion it supports a rollback. Promotion stays
by hand; this only reports.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import platform
from src.compare import daily_loss_difference, diebold_mariano

ALPHA = 0.05
MIN_DAYS = 14     # two of the test's 7 day lag windows; with fewer the variance can be 0 or negative
TARGET = "actual"
# The Hopsworks type of every gate_reports column, declared so a batch whose column is all None (model_mae,
# champion_mae, and first_day before any shared day) still inserts. status is added by the monitoring job.
SCHEMA = {"as_of": "timestamp", "model_version": "string", "champion_version": "string", "days": "bigint",
          "rows": "bigint", "model_mae": "double", "champion_mae": "double", "stat": "double", "p_one_sided": "double",
          "p_two_sided": "double", "first_day": "string", "alpha": "double", "status": "string"}


def one_sided_better(d: pd.Series) -> dict:
    """The one sided Diebold-Mariano test that a is better, on the daily loss difference d of
    src.compare.daily_loss_difference(a, b). With fewer than MIN_DAYS days, or identical forecasts, there is no
    evidence (p 1); a difference that never varies has no noise, so its sign decides."""
    if len(d) < MIN_DAYS or np.allclose(d, 0):
        return {"stat": 0.0, "p_one_sided": 1.0, "p_two_sided": 1.0}
    if np.ptp(d) == 0:
        better = float(d.iloc[0]) > 0
        return {"stat": 0.0, "p_one_sided": 0.0 if better else 1.0, "p_two_sided": 0.0}
    return diebold_mariano(d)


def live_gate(scored: pd.DataFrame, champion_version: str, shadow_version: str, cutoffs: dict[str, date],
              as_of: date) -> dict:
    """The gate_reports row (without the shadow's status) for one shadow model. scored: scored_predictions rows.
    cutoffs: the last training target date of each model, by version. The test needs MIN_DAYS shared
    days; before that p is 1."""
    after = max(cutoffs[champion_version], cutoffs[shadow_version])
    s = scored.assign(model_version=scored["model_version"].astype(str),
                      target_date=pd.to_datetime(scored["target_date"]))
    s = s[s["target_date"].dt.date > after]
    shadow = s[s["model_version"] == shadow_version]
    champ = s[s["model_version"] == champion_version]
    both = shadow[["city", "target_date"]].merge(champ[["city", "target_date"]], on=["city", "target_date"])
    report = {"as_of": as_of.isoformat(), "model_version": shadow_version, "champion_version": champion_version,
              "days": 0, "rows": 0, "model_mae": None, "champion_mae": None, "stat": 0.0, "p_one_sided": 1.0,
              "p_two_sided": 1.0, "first_day": None, "alpha": ALPHA}
    if both.empty:
        return report

    def rows(df):
        df = df.merge(both, on=["city", "target_date"])
        return pd.DataFrame({"row_id": df["city"] + "|" + df["target_date"].astype(str), "date": df["target_date"],
                             TARGET: df["actual"], "prediction": df["prediction"]})

    a, b = rows(shadow), rows(champ)
    d = daily_loss_difference(a, b, TARGET)
    report |= {k: v for k, v in one_sided_better(d).items() if k in ("stat", "p_one_sided", "p_two_sided")}
    return report | {"days": len(d), "rows": len(a),
                     "model_mae": float((a["prediction"] - a[TARGET]).abs().mean()),
                     "champion_mae": float((b["prediction"] - b[TARGET]).abs().mean()),
                     "first_day": both["target_date"].min().date().isoformat()}


def reports_group(fs):
    from hsfs.feature import Feature

    name, version = platform.GATE_REPORTS
    return fs.get_or_create_feature_group(
        name=name, version=version, primary_key=["as_of", "model_version"], event_time="as_of",
        online_enabled=False, features=[Feature(n, t) for n, t in SCHEMA.items()], description="Each shadow model against the champion on live days, by the monitoring job.")
