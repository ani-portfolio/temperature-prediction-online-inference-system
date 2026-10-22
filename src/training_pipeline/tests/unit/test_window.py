"""The training window: an expanding training range, then a holdout of the last labelled days every city has."""

from datetime import date

import pytest

from src.training_pipeline.window import TrainingWindow, training_window

START = date(2024, 1, 1)


def test_holdout_ends_on_the_last_day_every_city_has_labelled():
    w = training_window({"London": date(2026, 10, 21), "Tokyo": date(2026, 10, 20)}, START, holdout_days=28)
    assert w == TrainingWindow(train_start=START, cutoff=date(2026, 9, 22),
                               holdout_start=date(2026, 9, 23), holdout_end=date(2026, 10, 20))


def test_holdout_holds_exactly_holdout_days_and_follows_the_cutoff_with_no_gap():
    w = training_window({"London": date(2026, 10, 21)}, START, holdout_days=28)
    assert (w.holdout_end - w.holdout_start).days + 1 == 28
    assert (w.holdout_start - w.cutoff).days == 1


def test_no_training_days_left_is_an_error():
    with pytest.raises(ValueError, match="no training days"):
        training_window({"London": date(2024, 1, 28)}, START, holdout_days=28)


def test_no_labelled_cities_is_an_error():
    with pytest.raises(ValueError, match="no labelled"):
        training_window({}, START, holdout_days=28)
