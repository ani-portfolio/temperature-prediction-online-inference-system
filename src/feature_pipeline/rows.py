"""Feature rows for the feature store: one row per city and target date, built by the same build_features as training.

A row for target date T holds the inputs known at the end of day T minus the horizon (2 days), and the target once
day T is in the data. To serve tomorrow's forecast, the row for a date that has no weather yet is needed: it gets a
placeholder row with the city's static columns only, and build_features fills in its lags and rolling means from
the real days before it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.features import build_features, feature_columns


def _history_columns(config):
    """The columns that lags and rolling means read: never copied into a placeholder row."""
    cols = {config.get("target")}
    cols |= {lag["column"] for lag in config.get("lags") or []}
    cols |= {r["column"] for r in config.get("rolling") or []}
    return cols


def feature_rows(daily: pd.DataFrame, through: date, config: dict, entity: str = "city") -> pd.DataFrame:
    """Rows for every city and target date in daily, plus placeholder rows up to `through`.

    daily holds the raw daily rows (entity, the date column, and the source columns). Returns entity, target_date,
    the model inputs, and the target (empty for target dates that have no weather yet). `through` may be at most
    the horizon (min_shift_days) past each city's last day, since later rows would need days that are not known.
    """
    date_col, target = config["date_col"], config["target"]
    horizon = int(config.get("min_shift_days") or 0)
    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    through = pd.Timestamp(through)

    last = df.groupby(entity)[date_col].max()
    too_far = last[last + pd.Timedelta(days=horizon) < through]
    if len(too_far):
        raise ValueError(f"Rows up to {through.date()} need weather up to {(through - pd.Timedelta(days=horizon)).date()}"
                         f" (the horizon is {horizon} days), but {', '.join(too_far.index)} end earlier.")

    history = _history_columns(config)
    constant = [c for c in df.columns if c not in history and c not in (entity, date_col)
                and df.groupby(entity)[c].nunique(dropna=False).max() == 1]
    static = df.groupby(entity)[constant].first()
    future = [
        {entity: city, date_col: day, **static.loc[city].to_dict()}
        for city, end in last.items()
        for day in pd.date_range(end + pd.Timedelta(days=1), through, freq="D")
    ]
    full = pd.concat([df, pd.DataFrame(future, columns=[entity, date_col, *constant])], ignore_index=True) \
        if future else df

    built = build_features(full, config)
    inputs = [c for c in feature_columns(config) if c != entity]
    out = built[[entity, date_col, *inputs, target]].rename(columns={date_col: "target_date"})
    return out.sort_values([entity, "target_date"]).reset_index(drop=True)
