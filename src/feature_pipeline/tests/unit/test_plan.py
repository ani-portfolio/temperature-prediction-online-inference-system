"""What the scheduled run does for each city: fetch the days weather_daily lacks, and rebuild the feature rows that
weather_features lacks, so a run that wrote the days but not the features heals on the next run."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from src.feature_pipeline import pipeline
from src.feature_pipeline.pipeline import CityWork, plan_work

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)   # London's last complete day is 21 October
D = date(2026, 10, 21)
ZONE = {"London": "Europe/London"}


def plan(daily_newest, features_newest):
    return plan_work({"London": daily_newest}, {"London": features_newest}, ZONE, NOW, horizon=2)


def test_a_new_day_is_fetched_and_its_rows_built():
    assert plan(D - timedelta(days=1), D + timedelta(days=1)) == [
        CityWork("London", fetch_from=D, fetch_to=D, rebuild_from=D, rebuild_to=D + timedelta(days=2))]


def test_nothing_to_do_when_both_groups_are_current():
    assert plan(D, D + timedelta(days=2)) == []


def test_rows_missing_after_a_failed_run_are_rebuilt_without_fetching():
    # weather_daily got 21 October, but the run stopped before writing weather_features.
    assert plan(D, D + timedelta(days=1)) == [
        CityWork("London", fetch_from=None, fetch_to=None, rebuild_from=D, rebuild_to=D + timedelta(days=2))]


def test_missed_runs_are_caught_up():
    assert plan(D - timedelta(days=3), D - timedelta(days=1)) == [
        CityWork("London", fetch_from=D - timedelta(days=2), fetch_to=D, rebuild_from=D - timedelta(days=2),
                 rebuild_to=D + timedelta(days=2))]


def test_a_city_without_feature_rows_is_built_from_its_history():
    work = plan_work({"London": D}, {}, ZONE, NOW, horizon=2)
    assert work == [CityWork("London", fetch_from=None, fetch_to=None, rebuild_from=None,
                             rebuild_to=D + timedelta(days=2))]


# ---------------------------------------------------------------- run_latest heals end to end, with a fake store

class FakeGroup:
    def __init__(self, rows):
        self.rows = rows
        self.inserted = []

    def insert(self, df, wait=True):
        self.inserted.append(df)


def test_run_latest_rebuilds_rows_a_failed_run_left_out(monkeypatch):
    days = pd.date_range(end=pd.Timestamp(D), periods=420, freq="D")
    daily = pd.DataFrame({"city": "London", "country": "United Kingdom", "latitude": 51.5, "longitude": -0.1,
                          "grid_latitude": 51.5, "grid_longitude": -0.1, "elevation": 23.0,
                          "timezone": "Europe/London", "date": days.strftime("%Y-%m-%d"),
                          "temperature_2m_max": 15.0, "precipitation_sum": 1.0})
    config = {"numeric": ["latitude"], "categorical": ["city"], "date_col": "date", "target": "temperature_2m_max",
              "lags": [{"column": "temperature_2m_max", "days": [2], "by": "city"}], "min_shift_days": 2}
    monkeypatch.setattr(pipeline.platform, "login", lambda: type("P", (), {"get_feature_store": lambda s: None})())
    monkeypatch.setattr(pipeline.store, "daily_group", lambda fs: FakeGroup(None))
    monkeypatch.setattr(pipeline.store, "stored_daily", lambda fg, since: daily)
    monkeypatch.setattr(pipeline.store, "newest_feature_dates",
                        lambda fs: {"London": D + timedelta(days=1)})        # the failed run's rows are missing
    monkeypatch.setattr(pipeline, "fetch", lambda *a: (_ for _ in ()).throw(AssertionError("nothing to fetch")))
    monkeypatch.setattr(pipeline, "CITIES", [("London", "United Kingdom", 51.5, -0.1)])
    written = {}
    monkeypatch.setattr(pipeline, "_write", lambda fs, d, rows, cfg, rep: written.update(daily=d, rows=rows))

    report = pipeline.run_latest(now=NOW, config=config)
    assert report.problems == [] and report.stale_cities == []
    assert written["daily"].empty
    assert sorted(written["rows"]["target_date"].dt.date) == [D, D + timedelta(days=1), D + timedelta(days=2)]
    assert written["rows"].set_index("target_date").loc[pd.Timestamp(D), "temperature_2m_max"] == 15.0
