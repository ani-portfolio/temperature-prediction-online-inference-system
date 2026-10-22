"""The feature pipeline's two jobs: a one time backfill, and the hourly run that adds each city's newest days."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src import platform
from src.download_weather import CITIES, DAILY_VARIABLES
from src.feature_pipeline import store
from src.feature_pipeline.ingest import fetch, last_complete_day
from src.feature_pipeline.rows import feature_rows
from src.feature_pipeline.validate import STATIC, problems
from src.features import feature_columns, load_config

HISTORY_DAYS = 400   # covers the longest look back (the 364 day lag) plus the 28 day rolling mean


@dataclass
class FeatureRunReport:
    days_written: int = 0
    feature_rows_written: int = 0
    fresh_cities: list[str] = field(default_factory=list)
    stale_cities: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _spec():
    return yaml.safe_load((platform.ROOT / "ml_spec.yaml").read_text())


def _daily_types(df: pd.DataFrame) -> pd.DataFrame:
    """One schema for every insert: text keys, a timestamp date, and floats for every measure."""
    out = df[[*STATIC, "date", *DAILY_VARIABLES]].copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in ["latitude", "longitude", "grid_latitude", "grid_longitude", "elevation", *DAILY_VARIABLES]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def _feature_types(rows: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = rows.copy()
    for c in out.columns.drop(["city", "target_date"]):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def _write(fs, daily: pd.DataFrame, rows: pd.DataFrame, config: dict, report: FeatureRunReport):
    if len(daily):   # a run that only repairs feature rows has no new days
        fg_daily = store.daily_group(fs)
        store.ensure_suite(fg_daily)
        fg_daily.insert(_daily_types(daily), wait=True)
        report.days_written += len(daily)
    fg_rows = store.features_group(fs, ((_spec().get("features") or {}).get("code") or {}).get("features_hash"))
    fg_rows.insert(_feature_types(rows, config), wait=True)
    report.feature_rows_written += len(rows)
    inputs = [c for c in feature_columns(config) if c != "city"]
    store.feature_view(fs, fg_rows, inputs, config["target"])


def backfill(start: date, end: date | None = None, raw_csv: Path | None = None, config: dict | None = None,
             now: datetime | None = None) -> FeatureRunReport:
    """Write weather_daily and weather_features from start to end (default: the last day complete in every city).
    Days in the raw CSV come from it; later days are fetched. Nothing is written when a row breaks a rule."""
    config = config or load_config(root=platform.ROOT)
    report = FeatureRunReport()
    raw_csv = raw_csv or platform.ROOT / _spec()["split"]["data_source"]
    known = pd.read_csv(raw_csv)
    zones = known.groupby("city")["timezone"].first()
    end = end or min(last_complete_day(tz, now) for tz in zones)
    known = known[(known["date"] >= start.isoformat()) & (known["date"] <= end.isoformat())]
    fetched = pd.DataFrame()
    csv_end = date.fromisoformat(known["date"].max())
    if csv_end < end:
        fetched = fetch(CITIES, csv_end + timedelta(days=1), end)
    daily = pd.concat([known, fetched], ignore_index=True)
    report.problems = problems(daily)
    if report.problems:
        return report
    horizon = int(config.get("min_shift_days") or 0)
    rows = feature_rows(daily, end + timedelta(days=horizon), config)
    _write(platform.login().get_feature_store(), daily, rows, config, report)
    report.fresh_cities = sorted(daily["city"].unique())
    return report


@dataclass
class CityWork:
    """What one hourly run does for one city: fetch days fetch_from..fetch_to (None: nothing to fetch), then
    rebuild its feature rows for target dates rebuild_from..rebuild_to (rebuild_from None: from its first day)."""
    city: str
    fetch_from: date | None
    fetch_to: date | None
    rebuild_from: date | None
    rebuild_to: date


def plan_work(daily_newest: dict[str, date], features_newest: dict[str, date], zones: dict[str, str],
              now: datetime, horizon: int) -> list[CityWork]:
    """The work for each city, from the newest day in weather_daily and the newest target date in weather_features.

    Days are fetched when weather_daily is behind the city's last complete local day. Feature rows are rebuilt
    when weather_features is behind the stored days plus the horizon, also when nothing is fetched: a run that
    wrote the days but stopped before the features is then repaired by the next run. The rebuild starts at the
    first target date whose label or inputs may come from a day written after the features last were.
    """
    work = []
    for city in sorted(daily_newest):
        have, end = daily_newest[city], last_complete_day(zones[city], now)
        fetching = have < end
        fetch_from, fetch_to = (have + timedelta(days=1), end) if fetching else (None, None)
        rebuild_to = max(have, end) + timedelta(days=horizon)
        built = features_newest.get(city)
        if built is None:
            work.append(CityWork(city, fetch_from, fetch_to, None, rebuild_to))
        elif fetching or built < rebuild_to:
            rebuild_from = built - timedelta(days=horizon - 1)
            if fetching:
                rebuild_from = min(rebuild_from, fetch_from)
            work.append(CityWork(city, fetch_from, fetch_to, rebuild_from, rebuild_to))
    return work


def run_latest(now: datetime | None = None, config: dict | None = None) -> FeatureRunReport:
    """The hourly job: for each city, fetch the days after its newest stored day up to its last complete local
    day, and rebuild the feature rows those days change, plus any rows an earlier run failed to write (plan_work)."""
    config = config or load_config(root=platform.ROOT)
    now = now or datetime.now(UTC)
    horizon = int(config.get("min_shift_days") or 0)
    report = FeatureRunReport()
    fs = platform.login().get_feature_store()
    stored = store.stored_daily(store.daily_group(fs), now.date() - timedelta(days=HISTORY_DAYS + 3))
    newest = stored.groupby("city")["date"].max().map(date.fromisoformat)
    zones = stored.groupby("city")["timezone"].first()
    work = plan_work(newest.to_dict(), store.newest_feature_dates(fs), zones.to_dict(), now, horizon)

    tuples = {c[0]: c for c in CITIES}
    windows: dict[tuple[date, date], list] = {}
    for w in work:
        if w.fetch_from:
            windows.setdefault((w.fetch_from, w.fetch_to), []).append(tuples[w.city])
    new = [fetch(cities, start, end) for (start, end), cities in windows.items()]
    new = pd.concat(new, ignore_index=True) if new else pd.DataFrame(columns=stored.columns)
    report.problems = problems(new) if len(new) else []
    if report.problems:
        report.stale_cities = sorted(newest.index)
        return report

    daily = pd.concat([stored, new], ignore_index=True) if len(new) else stored
    rows = []
    for w in work:
        built = feature_rows(daily[daily["city"] == w.city], w.rebuild_to, config)
        rows.append(built if w.rebuild_from is None else built[built["target_date"] >= pd.Timestamp(w.rebuild_from)])
    if rows:
        _write(fs, new, pd.concat(rows, ignore_index=True), config, report)
    latest = daily.groupby("city")["date"].max().map(date.fromisoformat)
    for name in sorted(latest.index):
        fresh = latest[name] >= last_complete_day(zones[name], now)
        (report.fresh_cities if fresh else report.stale_cities).append(name)
    return report
