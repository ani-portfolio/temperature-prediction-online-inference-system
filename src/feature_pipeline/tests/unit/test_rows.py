"""feature_rows: the same features as training, one row per city and target date, with no lookahead."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline import feature_rows
from src.features import build_features, feature_columns

# A small version of the project's config: target lags from the horizon (2 days), a rolling mean, and calendar parts.
CONFIG = {
    "numeric": ["latitude", "elevation"],
    "categorical": ["city"],
    "date_parts": {"date": ["dayofyear", "month"]},
    "lags": [{"column": "temperature_2m_max", "days": [2, 3, 7], "by": "city"},
             {"column": "precipitation_sum", "days": [2], "by": "city"}],
    "rolling": [{"column": "temperature_2m_max", "windows": [7], "stats": ["mean"], "shift_days": 2, "by": "city"}],
    "date_col": "date",
    "target": "temperature_2m_max",
    "min_shift_days": 2,
    "target_min_shift_days": 2,
}
INPUTS = feature_columns(CONFIG)
LAST = date(2025, 2, 4)


def daily(days=400, end=LAST):
    """Raw daily rows for two cities, as ingestion writes them."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(end=pd.Timestamp(end), periods=days, freq="D")
    frames = []
    for city, lat, elev, tz in (("Oslo", 59.9, 23.0, "Europe/Oslo"), ("Lima", -12.0, 154.0, "America/Lima")):
        frames.append(pd.DataFrame({
            "city": city, "latitude": lat, "elevation": elev, "timezone": tz, "date": dates.strftime("%Y-%m-%d"),
            "temperature_2m_max": rng.normal(15, 5, days).round(1),
            "precipitation_sum": rng.gamma(1, 2, days).round(1),
        }))
    return pd.concat(frames, ignore_index=True)


def by_key(df):
    return df.set_index(["city", "target_date"]).sort_index()


def test_rows_match_build_features_on_the_same_days():
    raw = daily()
    expected = build_features(raw, CONFIG).rename(columns={"date": "target_date"})
    expected["target_date"] = pd.to_datetime(expected["target_date"])
    got = feature_rows(raw, LAST, CONFIG)
    assert list(got.columns) == ["city", "target_date", *[c for c in INPUTS if c != "city"], "temperature_2m_max"]
    cols = [c for c in INPUTS if c != "city"] + ["temperature_2m_max"]
    pd.testing.assert_frame_equal(by_key(got)[cols], by_key(expected)[cols], check_dtype=False)


@pytest.mark.parametrize("days_before", [0, 1])
def test_a_row_never_sees_its_own_day_or_the_day_before(days_before):
    raw = daily()
    target = pd.Timestamp(LAST) - pd.Timedelta(days=30)
    before = feature_rows(raw, LAST, CONFIG)
    changed = raw.copy()
    hit = pd.to_datetime(changed["date"]) == target - pd.Timedelta(days=days_before)
    changed.loc[hit, ["temperature_2m_max", "precipitation_sum"]] = 999.0
    after = feature_rows(changed, LAST, CONFIG)
    inputs = [c for c in INPUTS if c != "city"]
    row = lambda df: df[df["target_date"] == target].set_index("city")[inputs]
    pd.testing.assert_frame_equal(row(before), row(after))


def test_future_target_dates_get_rows_with_inputs_and_no_target():
    raw = daily()
    got = feature_rows(raw, date(2025, 2, 6), CONFIG)
    future = got[got["target_date"] > pd.Timestamp(LAST)]
    assert sorted(future["target_date"].dt.date.unique()) == [date(2025, 2, 5), date(2025, 2, 6)]
    assert set(future["city"]) == {"Oslo", "Lima"}
    assert future["temperature_2m_max"].isna().all()
    lag2 = future.set_index(["city", "target_date"])["temperature_2m_max_lag_2d"]
    last = raw[raw["date"] == "2025-02-04"].set_index("city")["temperature_2m_max"]
    assert lag2.loc[("Oslo", pd.Timestamp("2025-02-06"))] == last["Oslo"]
    assert future[["latitude", "elevation", "date_dayofyear", "temperature_2m_max_roll_7d_mean_shift_2d"]].notna().all().all()


def test_through_cannot_reach_past_the_horizon():
    with pytest.raises(ValueError, match="2 days"):
        feature_rows(daily(), date(2025, 2, 7), CONFIG)
