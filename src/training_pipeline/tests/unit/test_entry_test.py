"""The entry test: a candidate becomes the challenger only when a one sided Diebold-Mariano test says it beats the
reference (the champion, or persistence before there is one) on the holdout."""

import numpy as np
import pandas as pd
import pytest

from src.training_pipeline.gate import GateReport, entry_test

TARGET = "temperature_2m_max"


@pytest.fixture
def holdout():
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-09-23", periods=28)
    df = pd.DataFrame([(c, d) for c in ["London", "Tokyo", "Lima"] for d in days], columns=["city", "target_date"])
    df[TARGET] = 20 + rng.normal(0, 5, len(df))
    return df


def noisy(holdout, scale, seed):
    return holdout[TARGET].to_numpy() + np.random.default_rng(seed).normal(0, scale, len(holdout))


def test_a_clearly_better_candidate_becomes_the_challenger(holdout):
    status, dm = entry_test(holdout, noisy(holdout, 1, 1), noisy(holdout, 3, 2))
    assert status == "challenger"
    assert dm["p_one_sided"] < 0.05
    assert dm["candidate_mae"] < dm["reference_mae"]


def test_a_clearly_worse_candidate_is_rejected(holdout):
    status, dm = entry_test(holdout, noisy(holdout, 3, 1), noisy(holdout, 1, 2))
    assert status == "rejected"
    assert dm["p_two_sided"] < 0.05   # significantly worse, not just not better


def test_identical_forecasts_are_rejected(holdout):
    same = noisy(holdout, 2, 1)
    status, dm = entry_test(holdout, same, same.copy())
    assert status == "rejected"
    assert dm["p_one_sided"] == 1.0


def test_predictions_must_cover_every_holdout_row(holdout):
    with pytest.raises(ValueError, match="rows"):
        entry_test(holdout, noisy(holdout, 1, 1)[:-1], noisy(holdout, 3, 2))


def test_a_new_challenger_reminds_the_owner_to_restart_the_deployment():
    report = GateReport(7, "champion v3", 1.8, 2.0, {}, "challenger")
    assert "v7" in report.next_step and "temperatureforecast" in report.next_step
    assert GateReport(7, "champion v3", 2.1, 2.0, {}, "rejected").next_step == ""
