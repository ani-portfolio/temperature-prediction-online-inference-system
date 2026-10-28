"""Reference predictions: the last year of training days, predicted by a model fitted only on the days before it, so
the monitoring pipeline can compare live predictions with out-of-fold ones."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.training_pipeline.reference import REFERENCE_DAYS, reference_predictions

CUTOFF = date(2026, 9, 22)


def rows(first, last, cities=("London", "Tokyo")):
    days = pd.date_range(first, last, freq="D")
    X = pd.DataFrame([{"city": c, "target_date": d, "x": float(i)} for i, d in enumerate(days) for c in cities])
    return X, X["x"] * 2.0


class Recorder:
    """A model that remembers the target dates it was fitted on and predicts 2x."""

    def fit(self, X, y):
        self.seen = pd.to_datetime(X["target_date"]).dt.date
        return self

    def predict(self, X):
        return X["x"].to_numpy() * 2.0


def test_the_reference_is_the_last_364_training_days_predicted_out_of_fold():
    X, y = rows(date(2024, 1, 1), CUTOFF)
    made = []

    def build():
        made.append(Recorder())
        return made[-1]

    ref = reference_predictions(build, X, y, CUTOFF)
    start = CUTOFF - timedelta(days=REFERENCE_DAYS - 1)
    assert list(ref.columns) == ["city", "target_date", "prediction"]
    assert pd.to_datetime(ref["target_date"]).dt.date.min() == start
    assert pd.to_datetime(ref["target_date"]).dt.date.max() == CUTOFF
    assert len(ref) == REFERENCE_DAYS * 2
    assert made[0].seen.max() < start          # no reference day was in the refit
    np.testing.assert_allclose(ref["prediction"], X.loc[ref.index, "x"] * 2.0)


def test_too_little_history_before_the_reference_year_is_an_error():
    X, y = rows(CUTOFF - timedelta(days=REFERENCE_DAYS + 5), CUTOFF)
    with pytest.raises(ValueError, match="before the reference year"):
        reference_predictions(Recorder, X, y, CUTOFF)
