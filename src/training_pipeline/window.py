"""The dates a training run uses: an expanding training range, then a holdout for the entry test."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class TrainingWindow:
    train_start: date
    cutoff: date            # the last target date in training
    holdout_start: date
    holdout_end: date       # the last target date every city has labelled


def training_window(newest_labelled: dict[str, date], start: date, holdout_days: int) -> TrainingWindow:
    """Train on target dates from start to the cutoff, and hold out the holdout_days after it.

    The holdout ends on the earliest of the cities' newest labelled target dates, so every city is labelled on every
    holdout day. There is no gap between training and holdout, as in the split (gap_days 0): the 2 day horizon is
    already in the features.
    """
    if not newest_labelled:
        raise ValueError("There are no labelled rows: the feature store has no target dates with a label yet.")
    holdout_end = min(newest_labelled.values())
    holdout_start = holdout_end - timedelta(days=holdout_days - 1)
    cutoff = holdout_start - timedelta(days=1)
    if cutoff < start:
        raise ValueError(f"A {holdout_days} day holdout ending {holdout_end} leaves no training days after {start}.")
    return TrainingWindow(start, cutoff, holdout_start, holdout_end)
