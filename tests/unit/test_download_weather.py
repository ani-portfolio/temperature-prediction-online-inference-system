import urllib.parse

from src.download_weather import CITIES, DAILY_VARIABLES, build_url, to_frame


def test_city_names_are_unique():
    assert len({name for name, *_ in CITIES}) == len(CITIES) == 50


def test_build_url_lists_every_city_and_variable():
    batch = CITIES[:2]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(build_url(batch, "2024-01-01", "2024-01-31")).query)
    assert query["latitude"] == [f"{batch[0][2]},{batch[1][2]}"]
    assert query["daily"] == [",".join(DAILY_VARIABLES)]
    assert query["timezone"] == ["auto"]


def test_to_frame_adds_city_and_grid_columns():
    payload = {
        "latitude": 51.49, "longitude": -0.16, "elevation": 17.0, "timezone": "Europe/London",
        "daily": {"time": ["2024-01-01", "2024-01-02"], "temperature_2m_max": [12.2, 12.9]},
    }
    frame = to_frame(CITIES[0], payload)
    assert list(frame.columns[:9]) == [
        "city", "country", "latitude", "longitude", "grid_latitude", "grid_longitude", "elevation", "timezone", "date",
    ]
    assert frame["city"].unique().tolist() == ["London"]
    assert frame["temperature_2m_max"].tolist() == [12.2, 12.9]
