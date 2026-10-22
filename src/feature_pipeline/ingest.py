"""Raw daily weather from the Open-Meteo archive, by each city's local calendar day.

Reuses the request and parsing of src/download_weather.py, which built the training data, so live rows have the
same columns and meaning.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from src.download_weather import CITIES, build_url, fetch_json, to_frame


def last_complete_day(tz: str, now: datetime | None = None) -> date:
    """The last local calendar day that has ended in timezone tz."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(ZoneInfo(tz)).date() - timedelta(days=1)


def fetch(cities: list[tuple], start: date, end: date) -> pd.DataFrame:
    """Daily rows from start to end (inclusive) for the cities, as (name, country, latitude, longitude) tuples, in
    one request. Dates are each city's local days, as in the training data."""
    if not cities:
        return pd.DataFrame()
    payload = fetch_json(build_url(cities, start.isoformat(), end.isoformat()))
    payloads = payload if isinstance(payload, list) else [payload]
    return pd.concat([to_frame(city, p) for city, p in zip(cities, payloads)], ignore_index=True)


def city(name: str) -> tuple:
    """The (name, country, latitude, longitude) tuple of a city in the project."""
    return next(c for c in CITIES if c[0] == name)
