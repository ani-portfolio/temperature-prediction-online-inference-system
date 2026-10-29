"""Inference log messages from the deployment's Kafka topic become rows of the predictions feature group: one row per
model per answered city, keyed by request id and model."""

import base64
import io
import json
from pathlib import Path

import fastavro

from src.inference_pipeline.log import ENVELOPE, decode, log_rows

SPIKE = json.loads((Path(__file__).parent / "fixtures" / "inference_log_messages.json").read_text())


def message(kind, payload, inference_id="inf-1", code=200, ts=1792779509):
    out = io.BytesIO()
    fastavro.schemaless_writer(out, fastavro.parse_schema(ENVELOPE), {
        "modelId": 1, "modelName": "temperature_max_forecast", "modelVersion": 3, "requestTimestamp": ts,
        "responseHttpCode": code, "inferenceId": inference_id, "messageType": f"org.kubeflow.serving.inference.{kind}",
        "payload": json.dumps(payload)})
    return out.getvalue()


def answer(city="Auckland", request_id="req-1", version=3, reason=None, shadows=()):
    return {"city": city, "target_date": "2026-10-25", "forecast_max_c": 18.4, "model_version": version,
            "fallback_used": reason is not None, "fallback_reason": reason, "inputs_as_of": "2026-10-23",
            "log": {"request_id": request_id, "persistence_c": 17.0, "feature_row": "2026-10-25",
                    "shadows": list(shadows)}}


def test_decodes_the_messages_the_spike_captured():
    request, response = (decode(base64.b64decode(m)) for m in SPIKE["messages"])
    assert request["messageType"].endswith(".request")
    assert response["inferenceId"] == request["inferenceId"]
    assert json.loads(response["payload"])["predictions"][0]["log"]["shadows"] == [
        {"model_version": 4, "prediction": 19.0}]


def test_one_row_for_the_served_model_and_one_per_shadow():
    rows = log_rows([message("request", {"instances": [{"city": "Auckland"}]}),
                     message("response", {"predictions": [answer(shadows=[{"model_version": 4, "prediction": 19.0},
                                                                           {"model_version": 5, "prediction": None}])]})])
    assert list(rows["model_version"]) == ["3", "4", "5"]
    assert list(rows["role"]) == ["served", "shadow", "shadow"]
    assert list(rows["prediction"].fillna(-1)) == [18.4, 19.0, -1]
    first = rows.iloc[0]
    assert (first["request_id"], first["city"], first["persistence_c"]) == ("req-1", "Auckland", 17.0)
    assert str(first["target_date"].date()) == "2026-10-25"
    assert str(first["request_time"]) == "2026-10-23 18:18:29+00:00"
    assert rows["fallback_used"].tolist() == [False, False, False]


def test_a_fallback_is_logged_under_what_served_it():
    rows = log_rows([message("response", {"predictions": [answer(version="persistence", reason="stale")]})])
    assert rows[["model_version", "role", "fallback_reason"]].values.tolist() == [["persistence", "served", "stale"]]


def test_errors_requests_and_failed_responses_are_skipped():
    rows = log_rows([message("request", {"instances": [{"city": "Atlantis"}]}),
                     message("response", {"predictions": [{"city": "Atlantis", "error": "not served"}, answer()]}),
                     message("response", {"error": "boom"}, inference_id="inf-2", code=500)])
    assert rows["city"].tolist() == ["Auckland"]


def test_a_message_read_twice_is_kept_once():
    m = message("response", {"predictions": [answer()]})
    assert len(log_rows([m, m])) == 1


def test_no_messages_is_an_empty_table_with_the_columns():
    rows = log_rows([])
    assert rows.empty and {"request_id", "model_version", "request_time"} <= set(rows.columns)


def test_a_batch_with_no_fallback_still_has_a_typed_fallback_reason():
    # Hopsworks refuses to insert an all-null column, which pandas makes of a column of only None
    import pyarrow as pa

    rows = log_rows([message("response", {"predictions": [answer(shadows=[{"model_version": 4, "prediction": None}])]})])
    schema = pa.Schema.from_pandas(rows, preserve_index=False)
    assert [f.name for f in schema if pa.types.is_null(f.type)] == []
