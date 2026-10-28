"""The serving rules: what one request for a city gets, from the feature rows found and the models loaded. Pure: the
predictor does the lookups and calls this, so every rule is unit tested without Hopsworks.

A feature row for target date T holds inputs up to T minus 2 days, and its temperature_2m_max_lag_2d is the maximum
on T minus 2, which is persistence. The newest row with that value decides the path:

  the row for tomorrow                  the champion; persistence when it fails (model_error) or there is none
  1 day older                           persistence from that row (awaiting_today): the feature job runs once a day,
                                        so tomorrow's row is missing from local midnight until its next run
  2 or 3 days older                     persistence from that row (stale): a daily run was missed
  none of them                          the seasonal naive: the maximum 364 days before tomorrow, the same lag as the
                                        features, read as the lag_2d of the row for tomorrow minus 362 days
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.download_weather import TIMEZONES
from src.registry import ServingSet

CONTRACT = ("city", "target_date", "forecast_max_c", "model_version", "fallback_used", "fallback_reason",
            "inputs_as_of")
FALLBACK_REASONS = ("model_error", "stale", "awaiting_today", "seasonal", "no_champion")
PERSISTENCE = "temperature_2m_max_lag_2d"
HORIZON_DAYS = 2              # a row for T uses data up to T minus 2 (min_shift_days in ml_spec.yaml)
STALE_DAYS = 3                # rows older than this serve the seasonal naive instead of persistence
SEASONAL_ROW_DAYS = 362       # that row's lag_2d is the maximum 364 days before tomorrow


@dataclass
class Model:
    version: int
    pipeline: object          # a fitted pipeline: predict(X) on one feature row


def _local_now(city: str, now: datetime) -> datetime:
    if city not in TIMEZONES:
        raise KeyError(f"{city} is not one of the {len(TIMEZONES)} cities served.")
    return now.astimezone(ZoneInfo(TIMEZONES[city]))


def _tomorrow(city: str, now: datetime) -> date:
    return _local_now(city, now).date() + timedelta(days=1)


def lookup_keys(city: str, now: datetime) -> list[tuple[str, date]]:
    """The (city, target_date) keys to read from the online feature view in one call: tomorrow, the three days
    before it, and the seasonal row."""
    tomorrow = _tomorrow(city, now)
    days = [tomorrow - timedelta(days=k) for k in range(STALE_DAYS + 1)]
    return [(city, d) for d in [*days, tomorrow - timedelta(days=SEASONAL_ROW_DAYS)]]


def _value(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _predict(model: Model, row: dict) -> float | None:
    """The model's forecast for one row, or None when it fails or returns nothing usable."""
    try:
        return _value(model.pipeline.predict(pd.DataFrame([row]))[0])
    except Exception:   # noqa: BLE001 - any model failure falls back; the response says so
        return None


def forecast(city: str, now: datetime, rows: dict[date, dict], models: ServingSet, request_id: str) -> dict:
    """The response for one request. rows holds the feature rows found for lookup_keys, by target date; models holds
    the champion (None before the first promotion) and the shadows, as Model. Raises KeyError for a city that is not
    served and LookupError when no row gives a forecast."""
    tomorrow = _tomorrow(city, now)

    def response(value, version, reason, as_of, feature_row, persistence, shadows=()):
        return {"city": city, "target_date": tomorrow.isoformat(), "forecast_max_c": value, "model_version": version,
                "fallback_used": reason is not None, "fallback_reason": reason, "inputs_as_of": as_of.isoformat(),
                "log": {"request_id": request_id, "persistence_c": persistence,
                        "feature_row": feature_row and feature_row.isoformat(), "shadows": list(shadows)}}

    for k in range(STALE_DAYS + 1):
        day = tomorrow - timedelta(days=k)
        persistence = _value((rows.get(day) or {}).get(PERSISTENCE))
        if persistence is None:
            continue
        as_of = day - timedelta(days=HORIZON_DAYS)
        if k:
            return response(persistence, "persistence", "awaiting_today" if k == 1 else "stale", as_of, day,
                            persistence)
        shadows = [{"model_version": m.version, "prediction": _predict(m, rows[day])} for m in models.shadows]
        if models.champion is None:
            return response(persistence, "persistence", "no_champion", as_of, day, persistence, shadows)
        value = _predict(models.champion, rows[day])
        if value is None:
            return response(persistence, "persistence", "model_error", as_of, day, persistence, shadows)
        return response(value, models.champion.version, None, as_of, day, persistence, shadows)

    seasonal_day = tomorrow - timedelta(days=SEASONAL_ROW_DAYS)
    seasonal = _value((rows.get(seasonal_day) or {}).get(PERSISTENCE))
    if seasonal is None:
        raise LookupError(f"No feature row gives a forecast for {city} on {tomorrow}.")
    return response(seasonal, "seasonal_naive", "seasonal", seasonal_day - timedelta(days=HORIZON_DAYS),
                    seasonal_day, None)


def contract_problems(response: dict, city: str) -> list[str]:
    """What breaks the serving contract in one answer for city, or nothing. Every fallback the rules above give keeps
    the contract; a {city, error} answer does not."""
    if "error" in response:
        return [f"{city}: {response['error']}"]
    out = [f"{key} is missing" for key in CONTRACT if key not in response]
    if "city" in response and response["city"] != city:
        out.append(f"city is {response['city']!r}, not {city!r}")
    value = response.get("forecast_max_c")
    if not isinstance(value, (int, float)) or not -90 <= value <= 60:
        out.append(f"forecast_max_c {value!r} is not a temperature between -90 and 60 C")
    if response.get("fallback_used") and response.get("fallback_reason") not in FALLBACK_REASONS:
        out.append(f"fallback_reason {response.get('fallback_reason')!r} is not one of {', '.join(FALLBACK_REASONS)}")
    return out
