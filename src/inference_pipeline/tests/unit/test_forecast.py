"""The serving rules: which forecast a request gets from the feature rows found and the models loaded."""

from datetime import UTC, date, datetime

import pytest

from src.inference_pipeline.forecast import CONTRACT, Model, forecast, lookup_keys
from src.registry import ServingSet

PERSISTENCE = "temperature_2m_max_lag_2d"
NOON = datetime(2026, 10, 23, 23, 0, tzinfo=UTC)        # 12:00 on 2026-10-24 in Auckland (UTC+13)
AFTER_MIDNIGHT = datetime(2026, 10, 23, 11, 30, tzinfo=UTC)   # 00:30 on 2026-10-24 in Auckland
TOMORROW = date(2026, 10, 25)


class Fixed:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        assert len(X) == 1
        return [self.value]


class Broken:
    def predict(self, X):
        raise RuntimeError("boom")


def row(target_date, lag_2d=17.0):
    return {"city": "Auckland", "target_date": target_date, PERSISTENCE: lag_2d}


def fresh_rows():
    return {TOMORROW: row(TOMORROW, 17.0), date(2026, 10, 24): row(date(2026, 10, 24), 16.0)}


CHAMPION = Fixed(18.4)


def models(champion=CHAMPION, *shadows):
    return ServingSet(None if champion is None else Model(3, champion),
                      [Model(4 + i, s) for i, s in enumerate(shadows)])


def test_lookup_keys_are_tomorrow_to_three_days_back_and_the_seasonal_row():
    assert lookup_keys("Auckland", NOON) == [("Auckland", date(2026, 10, 25)), ("Auckland", date(2026, 10, 24)),
                                             ("Auckland", date(2026, 10, 23)), ("Auckland", date(2026, 10, 22)),
                                             ("Auckland", date(2025, 10, 28))]


def test_tomorrow_is_the_city_local_date():
    assert lookup_keys("Los Angeles", NOON)[0] == ("Los Angeles", date(2026, 10, 24))   # 16:00 on the 23rd there


def test_a_fresh_row_is_served_by_the_champion():
    r = forecast("Auckland", NOON, fresh_rows(), models(), "req-1")
    assert set(CONTRACT) <= set(r)
    assert r["target_date"] == "2026-10-25"
    assert r["forecast_max_c"] == 18.4
    assert r["model_version"] == 3
    assert r["fallback_used"] is False and r["fallback_reason"] is None
    assert r["inputs_as_of"] == "2026-10-23"
    assert r["log"] == {"request_id": "req-1", "persistence_c": 17.0, "feature_row": "2026-10-25",
                        "shadows": []}


def test_shadows_are_logged_and_never_served():
    r = forecast("Auckland", NOON, fresh_rows(), models(Fixed(18.4), Fixed(19.0), Broken()), "req-1")
    assert r["forecast_max_c"] == 18.4
    assert r["log"]["shadows"] == [{"model_version": 4, "prediction": 19.0}, {"model_version": 5, "prediction": None}]


def test_a_failing_champion_falls_back_to_persistence():
    r = forecast("Auckland", NOON, fresh_rows(), models(Broken()), "req-1")
    assert (r["forecast_max_c"], r["model_version"], r["fallback_reason"]) == (17.0, "persistence", "model_error")
    assert r["fallback_used"] is True


def test_no_champion_serves_persistence_and_still_runs_shadows():
    r = forecast("Auckland", NOON, fresh_rows(), models(None, Fixed(19.0)), "req-1")
    assert (r["forecast_max_c"], r["model_version"], r["fallback_reason"]) == (17.0, "persistence", "no_champion")
    assert r["log"]["shadows"] == [{"model_version": 4, "prediction": 19.0}]


def test_just_after_midnight_the_missing_row_is_awaited_not_stale():
    rows = {date(2026, 10, 24): row(date(2026, 10, 24), 16.0)}
    r = forecast("Auckland", AFTER_MIDNIGHT, rows, models(), "req-1")
    assert (r["forecast_max_c"], r["fallback_reason"], r["inputs_as_of"]) == (16.0, "awaiting_today", "2026-10-22")


@pytest.mark.parametrize("days_old", [1, 2, 3])
def test_a_row_up_to_three_days_old_serves_persistence(days_old):
    newest = date.fromordinal(TOMORROW.toordinal() - days_old)
    r = forecast("Auckland", NOON, {newest: row(newest, 15.0)}, models(), "req-1")
    assert (r["forecast_max_c"], r["model_version"], r["fallback_reason"]) == (15.0, "persistence", "stale")
    assert r["log"]["feature_row"] == newest.isoformat()


def test_older_inputs_serve_the_seasonal_naive():
    seasonal = date(2025, 10, 28)     # its lag_2d is the maximum on 2025-10-26, 364 days before tomorrow
    r = forecast("Auckland", NOON, {seasonal: row(seasonal, 14.0)}, models(), "req-1")
    assert (r["forecast_max_c"], r["model_version"], r["fallback_reason"]) == (14.0, "seasonal_naive", "seasonal")
    assert r["inputs_as_of"] == "2025-10-26"
    assert r["log"]["persistence_c"] is None


def test_a_row_without_the_last_day_maximum_does_not_count():
    rows = {TOMORROW: row(TOMORROW, None), date(2026, 10, 24): row(date(2026, 10, 24), 16.0)}
    r = forecast("Auckland", NOON, rows, models(), "req-1")
    assert (r["forecast_max_c"], r["fallback_reason"]) == (16.0, "stale")


def test_nothing_to_serve_is_an_error():
    with pytest.raises(LookupError, match="Auckland"):
        forecast("Auckland", NOON, {}, models(), "req-1")


def test_an_unknown_city_is_an_error():
    with pytest.raises(KeyError, match="Atlantis"):
        lookup_keys("Atlantis", NOON)
    with pytest.raises(KeyError, match="Atlantis"):
        forecast("Atlantis", NOON, {}, models(), "req-1")
