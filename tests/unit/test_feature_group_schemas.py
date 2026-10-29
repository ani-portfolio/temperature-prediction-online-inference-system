"""Every feature group the pipelines write declares its schema, so a batch whose column is all None still inserts.

Without a declared type, Hopsworks infers each column's type from the frame and refuses an all-None column
("dtype 'null' not supported"), which failed every inference_pipeline_log run on 2026-10-28. These tests run the
hsfs client's own type inference and pre-insert validation, without a connection, on frames the pipelines build.
"""

import io
import json
from datetime import date
from types import SimpleNamespace

import fastavro
import pandas as pd
import pytest
from hsfs.core.schema_validation import PandasValidator
from hsfs.engine.python import Engine

from src.inference_pipeline import log
from src.monitoring_pipeline import checks, gate, scores

ENGINE = Engine.__new__(Engine)   # only the schema parsing, which needs no client
AS_OF = date(2026, 10, 29)


class FeatureStore:
    def get_or_create_feature_group(self, **kwargs):
        return SimpleNamespace(**kwargs)


def predictions(reason):
    """log_rows on one inference log response, as the log job decodes it from Kafka."""
    out = io.BytesIO()
    payload = {"predictions": [{
        "city": "Rome", "target_date": "2026-10-30", "forecast_max_c": 19.7, "model_version": 2,
        "fallback_used": reason is not None, "fallback_reason": reason, "inputs_as_of": "2026-10-28",
        "log": {"request_id": "req-1", "persistence_c": 18.9, "feature_row": "2026-10-30", "shadows": []}}]}
    fastavro.schemaless_writer(out, fastavro.parse_schema(log.ENVELOPE), {
        "modelId": 1, "modelName": "temperature_max_forecast", "modelVersion": 2, "requestTimestamp": 1793318280,
        "responseHttpCode": 200, "inferenceId": "inf-1", "messageType": "org.kubeflow.serving.inference.response",
        "payload": json.dumps(payload)})
    return log.log_rows([out.getvalue()])


def scored():
    p = predictions(None)
    actuals = pd.DataFrame({"city": ["Rome"], "date": [pd.Timestamp("2026-10-30")], scores.TARGET: [20.1]})
    return scores.score(p, actuals, pd.DataFrame(columns=scores.KEY), pd.Timestamp("2026-10-31 13:30", tz="UTC"))


def gate_reports(with_days):
    s = pd.DataFrame()
    if with_days:
        days = pd.date_range("2026-10-01", periods=20)
        s = pd.DataFrame({"city": "Rome", "target_date": list(days) * 2, "model_version": ["2"] * 20 + ["3"] * 20,
                          "actual": 20.0, "prediction": [21.0] * 20 + [20.5] * 20})
    else:
        s = pd.DataFrame(columns=["city", "target_date", "model_version", "actual", "prediction"])
    cutoffs = {"2": date(2026, 9, 1), "3": date(2026, 9, 1)}
    row = gate.live_gate(s, "2", "3", cutoffs, AS_OF) | {"status": "challenger"}
    return pd.DataFrame([row]).assign(as_of=lambda d: pd.to_datetime(d["as_of"]))   # as run_daily inserts it


def check_reports(all_none):
    if all_none:   # every optional column empty: value, threshold, and note
        rows = [checks._row(AS_OF, "input_psi", "latitude")]
    else:
        rows = [checks._row(AS_OF, "input_psi", "latitude", 0.1, 0.25, False, 50, "a note")]
    return pd.DataFrame(rows, columns=checks.CHECK_COLUMNS).assign(as_of=lambda d: pd.to_datetime(d["as_of"]))


# (the group, a frame with every optional column filled, a frame with some columns all None)
GROUPS = {
    "predictions": (log._predictions_group, predictions("stale"), predictions(None)),
    "scored_predictions": (scores.scored_group, scored(), scored().assign(persistence_c=None,
                                                                          persistence_abs_error=None)),
    "gate_reports": (gate.reports_group, gate_reports(True), gate_reports(False)),
    "monitoring_reports": (checks.reports_group, check_reports(False), check_reports(True)),
}


def declared(make_group):
    return make_group(FeatureStore())


@pytest.mark.parametrize("name", GROUPS)
def test_the_group_declares_a_type_for_every_column_it_is_given(name):
    make_group, full, _ = GROUPS[name]
    group = declared(make_group)
    assert group.name == name
    assert [f.name for f in group.features] == list(full.columns)


@pytest.mark.parametrize("name", GROUPS)
def test_the_declared_types_are_the_ones_hopsworks_infers_from_a_full_batch(name):
    make_group, full, _ = GROUPS[name]
    inferred = {f.name: f.type for f in ENGINE._parse_schema_feature_group(full)}
    assert {f.name: f.type for f in declared(make_group).features} == inferred


@pytest.mark.parametrize("name", GROUPS)
def test_a_batch_with_an_all_none_column_passes_the_insert_checks(name):
    make_group, _, sparse = GROUPS[name]
    assert any(sparse[c].isna().all() for c in sparse.columns)
    group = declared(make_group)
    with pytest.raises(Exception, match="null"):
        ENGINE._parse_schema_feature_group(sparse)   # what failed in production
    parsed = ENGINE._parse_schema_feature_group(sparse, features=group.features)
    assert [f.type for f in parsed] == [f.type for f in group.features]
    errors, _, pk_null, too_long = PandasValidator()._validate_df_specifics(
        SimpleNamespace(primary_key=group.primary_key, columns=group.features), sparse)
    assert (errors, pk_null, too_long) == ({}, False, False)
