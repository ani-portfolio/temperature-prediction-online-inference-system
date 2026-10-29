"""The daily checks: drift against the same calendar days in training, live MAE against persistence, and the share of
requests the model answered."""

from datetime import date

import numpy as np
import pandas as pd

from src.monitoring_pipeline.checks import (
    CHECK_COLUMNS,
    IMPLAUSIBLE_C,
    canary_row,
    drift_rows,
    performance_row,
    served_share_row,
    skipped_rows,
)

AS_OF = date(2026, 10, 27)
RNG = np.random.default_rng(0)


def frame(first, days, cities=("London", "Tokyo"), shift=0.0, **cols):
    dates = pd.date_range(first, periods=days, freq="D")
    df = pd.DataFrame([{"city": c, "target_date": d} for d in dates for c in cities])
    for name in cols or ["x"]:
        df[name] = RNG.normal(shift, 1.0, len(df))
    return df


def test_input_drift_compares_only_the_same_calendar_days_in_training():
    ref = pd.concat([frame("2025-09-30", 28), frame("2025-03-01", 28, shift=5.0)])   # March is far off
    out = drift_rows(ref, frame("2026-09-30", 28), None, None, AS_OF)
    row = out[out["check"] == "input_psi"].iloc[0]
    assert row["subject"] == "x"
    assert row["value"] < 0.25 and not row["flagged"]
    assert row["rows"] == 56


def test_a_shifted_input_is_flagged():
    out = drift_rows(frame("2025-09-30", 28), frame("2026-09-30", 28, shift=2.0), None, None, AS_OF)
    assert out.set_index("check").loc["input_psi", "flagged"]


def test_prediction_drift_uses_only_the_cities_seen_live():
    ref = pd.concat([frame("2025-09-30", 28, cities=("London",)),
                     frame("2025-09-30", 28, cities=("Tokyo",), shift=6.0)]).rename(columns={"x": "prediction"})
    live = frame("2026-09-30", 28, cities=("London",)).rename(columns={"x": "prediction"})
    live["prediction"] = ref.loc[ref["city"] == "London", "prediction"].to_numpy()   # London has not moved
    out = drift_rows(frame("2025-09-30", 28), frame("2026-09-30", 28), ref, live, AS_OF)
    row = out.set_index("check").loc["prediction_psi"]
    assert not row["flagged"] and row["rows"] == 28


def test_no_reference_predictions_is_a_row_with_a_note_and_no_flag():
    out = drift_rows(frame("2025-09-30", 28), frame("2026-09-30", 28), None, frame("2026-09-30", 28), AS_OF)
    row = out.set_index("check").loc["prediction_psi"]
    assert row["note"] == "no_reference" and pd.isna(row["value"]) and not row["flagged"]
    assert list(out.columns) == CHECK_COLUMNS


def scored(errors, persistence_errors, version="3", first="2026-09-30"):
    days = pd.date_range(first, periods=len(errors), freq="D")
    return pd.DataFrame({"city": "London", "target_date": days, "model_version": version, "role": "served",
                         "prediction": 10.0 + np.asarray(errors), "actual": 10.0,
                         "persistence_c": 10.0 + np.asarray(persistence_errors)})


def test_a_model_better_than_persistence_is_not_flagged():
    r = performance_row(scored(RNG.normal(1.0, 0.2, 28), RNG.normal(3.0, 0.2, 28)), "3", AS_OF)
    assert r["check"] == "performance" and not r["flagged"]
    assert abs(r["value"] - 1.0) < 0.2 and r["rows"] == 28


def test_a_model_no_better_than_persistence_is_flagged():
    errors = RNG.normal(2.0, 0.5, 28)
    assert performance_row(scored(errors, errors + RNG.normal(0, 0.1, 28)), "3", AS_OF)["flagged"]


def test_performance_uses_the_last_28_days_and_the_served_model_only():
    s = pd.concat([scored([9.0] * 10, [0.0] * 10, first="2026-08-01"), scored([1.0] * 28, [3.0] * 28),
                   scored([5.0] * 28, [3.0] * 28, version="4")])
    r = performance_row(s, "3", AS_OF)
    assert (r["rows"], r["value"]) == (28, 1.0)


def logged(reason=None, at="2026-10-27 13:00"):
    return {"model_version": "persistence" if reason else "3", "role": "served", "fallback_reason": reason,
            "request_time": pd.Timestamp(at, tz="UTC")}


def test_served_share_leaves_out_awaiting_today_and_flags_under_95_percent():
    rows = [logged()] * 18 + [logged("stale")] * 2 + [logged("awaiting_today")] * 5 + [logged(at="2026-10-26 13:00")]
    r = served_share_row(pd.DataFrame(rows), AS_OF)
    assert (r["value"], r["rows"], r["flagged"]) == (0.9, 20, True)


def test_skipped_rows_say_why():
    out = skipped_rows(["performance", "prediction_psi"], AS_OF, "no_champion")
    assert out["note"].tolist() == ["no_champion", "no_champion"] and not out["flagged"].any()


def answered(city, target_date, arm, served, served_pred, shadows=(), reason=None, at="2026-10-26 10:00",
             persistence=17.0):
    """The predictions rows of one answered request: the served row and one per shadow, all with the city-day's arm."""
    common = {"request_id": f"{city}-{target_date}-{at}", "city": city, "target_date": pd.Timestamp(target_date),
              "request_time": pd.Timestamp(at, tz="UTC"), "arm": arm, "persistence_c": persistence,
              "fallback_reason": reason}
    return [{**common, "model_version": served, "role": "served", "prediction": served_pred},
            *({**common, "model_version": v, "role": "shadow", "prediction": p} for v, p in shadows)]


def canary(rows, since=None):
    return canary_row(pd.DataFrame([r for req in rows for r in req]), "9", "canary", 0.1, since, AS_OF)


def test_a_healthy_candidate_has_no_breaches():
    r = canary([answered("London", "2026-10-26", "candidate", "9", 18.0, [("3", 17.5)]),
                answered("Tokyo", "2026-10-26", "candidate", "9", 21.0, [("3", 20.0)]),
                answered("Lima", "2026-10-26", "champion", "3", 19.0, [("9", 19.5)])])
    assert (r["candidate_city_days"], r["checked_city_days"], r["breaches"], r["flagged"]) == (2, 3, 0, False)
    assert (r["model_version"], r["status"], r["traffic"], r["threshold_c"]) == ("9", "canary", 0.1, IMPLAUSIBLE_C)


def test_a_forecast_far_from_persistence_is_a_breach_counted_once_per_city_day():
    far = 17.0 + IMPLAUSIBLE_C + 1
    r = canary([answered("London", "2026-10-26", "candidate", "9", far, [("3", 17.5)]),
                answered("London", "2026-10-26", "candidate", "9", far, [("3", 17.5)], at="2026-10-26 11:00"),
                answered("Lima", "2026-10-26", "champion", "3", 19.0, [("9", -300.0)])])   # Kelvin gone wrong
    assert (r["implausible"], r["missing"], r["breaches"], r["flagged"]) == (2, 0, 2, True)


def test_a_candidate_failing_on_its_own_city_day_is_missing():
    # the caller got persistence; the candidate has no row of its own
    r = canary([answered("London", "2026-10-26", "candidate", "persistence", 17.0, [("3", 17.5)],
                         reason="model_error")])
    assert (r["candidate_city_days"], r["missing"], r["breaches"], r["flagged"]) == (1, 1, 1, True)


def test_a_candidate_failing_in_shadow_is_missing():
    r = canary([answered("Lima", "2026-10-26", "champion", "3", 19.0, [("9", None)])])
    assert (r["missing"], r["flagged"]) == (1, True)


def test_a_champion_failure_is_not_the_candidate_s_breach():
    r = canary([answered("Lima", "2026-10-26", "champion", "persistence", 17.0, [("9", 18.0)], reason="model_error")])
    assert (r["missing"], r["breaches"]) == (0, 0)


def test_only_requests_since_the_step_began_count():
    r = canary([answered("London", "2026-10-20", "candidate", "9", 60.0, at="2026-10-19 10:00"),
                answered("London", "2026-10-26", "candidate", "9", 18.0)], since=date(2026, 10, 25))
    assert (r["candidate_city_days"], r["breaches"], r["since"]) == (1, 0, "2026-10-25")


def test_no_candidate_rows_is_a_row_with_zeros():
    r = canary_row(pd.DataFrame(columns=["request_time", "city", "target_date", "model_version", "role", "arm",
                                         "prediction", "persistence_c", "fallback_reason"]),
                   "9", "canary", 0.1, None, AS_OF)
    assert (r["candidate_city_days"], r["checked_city_days"], r["breaches"], r["flagged"]) == (0, 0, 0, False)
