"""Each city's last complete local day, and the checks run on raw daily rows before they are written."""

from datetime import UTC, date, datetime

import pandas as pd

from src.feature_pipeline.ingest import last_complete_day
from src.feature_pipeline.validate import problems


def test_last_complete_day_is_local_to_each_city():
    now = datetime(2026, 10, 21, 13, 30, tzinfo=UTC)   # 02:30 on the 22nd in Auckland, 06:30 on the 21st in LA
    assert last_complete_day("Pacific/Auckland", now) == date(2026, 10, 21)
    assert last_complete_day("America/Los_Angeles", now) == date(2026, 10, 20)
    assert last_complete_day("Europe/London", now) == date(2026, 10, 20)


def good_rows():
    return pd.DataFrame({
        "city": ["London", "London", "Paris"],
        "country": ["United Kingdom", "United Kingdom", "France"],
        "latitude": [51.5, 51.5, 48.9], "longitude": [-0.1, -0.1, 2.4],
        "grid_latitude": [51.5, 51.5, 48.8], "grid_longitude": [-0.1, -0.1, 2.3],
        "elevation": [23.0, 23.0, 43.0], "timezone": ["Europe/London", "Europe/London", "Europe/Paris"],
        "date": ["2026-10-19", "2026-10-20", "2026-10-20"],
        "temperature_2m_max": [15.2, 14.1, 16.0], "temperature_2m_min": [8.0, 7.5, 9.1],
        "temperature_2m_mean": [11.4, 10.6, 12.2], "apparent_temperature_max": [13.0, 12.5, 15.1],
        "apparent_temperature_min": [5.0, 4.8, 7.0], "precipitation_sum": [0.0, 2.3, 0.4],
        "rain_sum": [0.0, 2.3, 0.4], "snowfall_sum": [0.0, 0.0, 0.0], "wind_speed_10m_max": [18.0, 25.1, 12.0],
        "wind_gusts_10m_max": [35.0, 48.2, 22.0], "wind_direction_10m_dominant": [240, 255, 190],
        "shortwave_radiation_sum": [8.1, 5.2, 9.0], "sunshine_duration": [21000.0, 9000.0, 25000.0],
        "weather_code": [3, 61, 2],
    })


def test_good_rows_have_no_problems():
    assert problems(good_rows(), known_cities={"London", "Paris"}) == []


def test_problems_name_each_kind_of_bad_row():
    bad = good_rows()
    bad.loc[0, "temperature_2m_max"] = 75.0                  # out of range
    bad.loc[1, "temperature_2m_min"] = 20.0                  # min above max
    bad = pd.concat([bad, bad.iloc[[2]],                       # Paris on 2026-10-20 twice
                     bad.iloc[[2]].assign(city="Atlantis")], ignore_index=True)
    found = problems(bad, known_cities={"London", "Paris"})
    text = " | ".join(found)
    assert "temperature_2m_max" in text and "75" in text, found
    assert "temperature_2m_min" in text and "temperature_2m_max" in text, found
    assert "Atlantis" in text, found
    assert "Paris" in text and "2026-10-20" in text, found


def test_missing_columns_are_a_problem():
    assert any("weather_code" in p for p in problems(good_rows().drop(columns="weather_code"),
                                                      known_cities={"London", "Paris"}))
