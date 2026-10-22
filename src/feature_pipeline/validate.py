"""Checks on raw daily rows before they become features: the columns, value ranges, consistent temperatures, one
row per city and date, and only the project's cities.

The same rules are attached to the weather_daily feature group as a Great Expectations suite (expectation_suite),
so the platform rejects a bad batch too; problems() runs them locally first, with readable messages.
"""

from __future__ import annotations

import pandas as pd

from src.download_weather import CITIES, DAILY_VARIABLES

STATIC = ["city", "country", "latitude", "longitude", "grid_latitude", "grid_longitude", "elevation", "timezone"]
COLUMNS = STATIC + ["date"] + DAILY_VARIABLES
TEMPERATURES = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_max",
                "apparent_temperature_min"]
RANGES = {
    **{c: (-90, 60) for c in TEMPERATURES},
    "precipitation_sum": (0, 1000), "rain_sum": (0, 1000), "snowfall_sum": (0, 1000),
    "wind_speed_10m_max": (0, 400), "wind_gusts_10m_max": (0, 500), "wind_direction_10m_dominant": (0, 360),
    "shortwave_radiation_sum": (0, 50), "sunshine_duration": (0, 86400), "weather_code": (0, 99),
    "latitude": (-90, 90), "longitude": (-180, 180),
}
KEYS = ["city", "date"]


def problems(daily: pd.DataFrame, known_cities: set[str] | None = None) -> list[str]:
    """Every rule the rows break, as readable sentences. Empty when the rows can be written."""
    known = known_cities if known_cities is not None else {c[0] for c in CITIES}
    out = []
    missing = [c for c in COLUMNS if c not in daily.columns]
    if missing:
        return [f"Columns missing: {', '.join(missing)}."]
    for col, (low, high) in RANGES.items():
        v = pd.to_numeric(daily[col], errors="coerce")
        bad = daily[(v < low) | (v > high)]
        for r in bad.itertuples():
            out.append(f"{col} is {getattr(r, col)} for {r.city} on {r.date}, outside {low} to {high}.")
    order = daily[(daily["temperature_2m_min"] > daily["temperature_2m_mean"])
                  | (daily["temperature_2m_mean"] > daily["temperature_2m_max"])]
    for r in order.itertuples():
        out.append(f"temperature_2m_min ({r.temperature_2m_min}), temperature_2m_mean ({r.temperature_2m_mean}) and "
                   f"temperature_2m_max ({r.temperature_2m_max}) are out of order for {r.city} on {r.date}.")
    dup = daily[daily.duplicated(KEYS, keep=False)].drop_duplicates(KEYS)
    for r in dup.itertuples():
        out.append(f"{r.city} on {r.date} appears more than once.")
    unknown = sorted(set(daily["city"]) - known)
    if unknown:
        out.append(f"Cities that are not in the project: {', '.join(unknown)}.")
    return out


def expectations() -> list[tuple[str, dict]]:
    """The rules as Great Expectations expectations: (expectation type, its arguments)."""
    out = [("expect_column_to_exist", {"column": c}) for c in COLUMNS]
    out += [("expect_column_values_to_be_between", {"column": c, "min_value": low, "max_value": high})
            for c, (low, high) in RANGES.items()]
    out.append(("expect_compound_columns_to_be_unique", {"column_list": KEYS}))
    out.append(("expect_column_values_to_be_in_set", {"column": "city", "value_set": sorted(c[0] for c in CITIES)}))
    return out


def expectation_suite():
    """The rules as the Hopsworks expectation suite of the weather_daily feature group."""
    from hsfs.expectation_suite import ExpectationSuite
    from hsfs.ge_expectation import GeExpectation

    return ExpectationSuite(expectation_suite_name="weather_daily", meta={},
                            expectations=[GeExpectation(expectation_type=t, kwargs=kw, meta={})
                                          for t, kw in expectations()])
