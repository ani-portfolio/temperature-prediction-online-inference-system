"""Reference predictions saved with each model, for the monitoring pipeline's prediction drift check.

The last REFERENCE_DAYS training days are predicted by a second fit of the same recipe on only the days before them,
so the reference is out-of-fold, like live predictions, rather than the final model predicting rows it was fitted on.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

REFERENCE_DAYS = 364     # a full year, so every calendar day has a reference
MIN_FIT_DAYS = 364       # the refit needs a full year before the reference year, to have seen every season


def reference_predictions(build, X: pd.DataFrame, y: pd.Series, cutoff: date) -> pd.DataFrame:
    """city, target_date, and prediction for the REFERENCE_DAYS training days ending on cutoff, from build() fitted
    on the rows before them. build returns an unfitted model."""
    start = cutoff - timedelta(days=REFERENCE_DAYS - 1)
    days = pd.to_datetime(X["target_date"]).dt.date
    fit, ref = (days < start).to_numpy(), ((days >= start) & (days <= cutoff)).to_numpy()
    if days[fit].nunique() < MIN_FIT_DAYS:
        raise ValueError(f"Only {days[fit].nunique()} training days before the reference year (from {start}); "
                         f"the out-of-fold refit needs {MIN_FIT_DAYS}.")
    model = build().fit(X[fit], y[fit])
    out = X.loc[ref, ["city", "target_date"]].copy()
    out["prediction"] = model.predict(X[ref])
    return out
