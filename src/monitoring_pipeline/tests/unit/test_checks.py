"""The daily checks: drift against the same calendar days in training, live MAE against persistence, and the share of
requests the model answered."""

from datetime import date

import numpy as np
import pandas as pd

from src.monitoring_pipeline.checks import CHECK_COLUMNS, drift_rows, performance_row, served_share_row, skipped_rows

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
