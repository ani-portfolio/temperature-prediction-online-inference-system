"""Label feedback: logged model predictions joined to the actual maximum, one row per city, target date, and model."""

import pandas as pd
import pytest

from src.monitoring_pipeline.scores import COLUMNS, score

NOW = pd.Timestamp("2026-10-28 13:30", tz="UTC")


def logged(city="London", target="2026-10-26", version="3", role="served", prediction=15.0, at="2026-10-25 13:00",
           persistence=14.0):
    return {"request_id": f"{city}-{at}", "model_version": version, "request_time": pd.Timestamp(at, tz="UTC"),
            "city": city, "target_date": pd.Timestamp(target), "role": role, "prediction": prediction,
            "fallback_used": version == "persistence", "fallback_reason": None, "inputs_as_of": None,
            "feature_row": None, "persistence_c": persistence}


def actual(city="London", day="2026-10-26", value=16.0):
    return {"city": city, "date": pd.Timestamp(day), "temperature_2m_max": value}


def run(predictions, actuals, scored=()):
    already = pd.DataFrame(list(scored), columns=["city", "target_date", "model_version"])
    return score(pd.DataFrame(predictions), pd.DataFrame(actuals), already, NOW)


def test_a_prediction_with_an_actual_is_scored_with_both_errors():
    out = run([logged()], [actual()])
    assert list(out.columns) == COLUMNS
    row = out.iloc[0]
    assert (row["actual"], row["abs_error"], row["persistence_abs_error"]) == (16.0, 1.0, 2.0)
    assert row["scored_at"] == NOW


def test_only_the_first_request_per_city_target_date_and_model_counts():
    out = run([logged(prediction=15.0, at="2026-10-25 13:00"), logged(prediction=99.0, at="2026-10-25 14:00")],
              [actual()])
    assert out["prediction"].tolist() == [15.0]


def test_fallback_rows_are_left_out_and_shadows_kept():
    out = run([logged(), logged(version="persistence", prediction=14.0, at="2026-10-25 05:00"),
               logged(version="4", role="shadow", prediction=15.5)], [actual()])
    assert sorted(out["model_version"]) == ["3", "4"]
    assert out.set_index("model_version").loc["4", "role"] == "shadow"


def test_a_prediction_without_an_actual_waits():
    assert run([logged(target="2026-10-28")], [actual()]).empty


def test_rows_already_scored_are_skipped():
    out = run([logged(), logged(city="Tokyo")], [actual(), actual(city="Tokyo")],
              scored=[("London", pd.Timestamp("2026-10-26"), "3")])
    assert out["city"].tolist() == ["Tokyo"]


@pytest.mark.parametrize("value", [None, float("nan")])
def test_a_missing_actual_value_waits(value):
    assert run([logged()], [actual(value=value)]).empty
