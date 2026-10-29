"""The log job: read the deployment's inference log topic and write one row per model per answered city to the
feature group predictions, which label feedback joins to the actuals.

The KServe logger writes two messages per request, the request and the response, each an Avro record with no schema
registered on the topic (the envelope below, found by the logging spike; see docs/deployment.md). Only responses are
needed: the predictor puts everything worth logging, shadows included, in the response's log block.
"""

from __future__ import annotations

import io
import json
import time

import fastavro
import pandas as pd

from src import platform

ENVELOPE = {"type": "record", "name": "InferenceLog", "fields": [
    {"name": "modelId", "type": "int"}, {"name": "modelName", "type": "string"},
    {"name": "modelVersion", "type": "int"}, {"name": "requestTimestamp", "type": "long"},   # Unix seconds
    {"name": "responseHttpCode", "type": "int"}, {"name": "inferenceId", "type": "string"},
    {"name": "messageType", "type": "string"}, {"name": "payload", "type": "string"}]}
_PARSED = fastavro.parse_schema(ENVELOPE)
# The Hopsworks type of every column, declared so a batch whose column is all None (fallback_reason, when the model
# served every answer) still inserts: without it the type is inferred from the batch, and None has none.
SCHEMA = {"request_id": "string", "model_version": "string", "request_time": "timestamp", "city": "string",
          "target_date": "timestamp", "role": "string", "prediction": "double", "fallback_used": "boolean",
          "fallback_reason": "string", "inputs_as_of": "timestamp", "feature_row": "timestamp",
          "persistence_c": "double", "arm": "string"}
COLUMNS = list(SCHEMA)
KEY = ["request_id", "model_version"]
CONSUMER_GROUP = "inference_pipeline_log"
IDLE_SECONDS = 30      # stop reading when the topic has been quiet this long


def decode(value: bytes) -> dict:
    return fastavro.schemaless_reader(io.BytesIO(value), _PARSED)


def _rows(envelope: dict) -> list[dict]:
    if not envelope["messageType"].endswith(".response") or envelope["responseHttpCode"] != 200:
        return []
    when = pd.Timestamp(envelope["requestTimestamp"], unit="s", tz="UTC")
    out = []
    for p in json.loads(envelope["payload"]).get("predictions") or []:
        if "error" in p:
            continue
        log = p.get("log") or {}
        common = {"request_id": log.get("request_id"), "request_time": when, "city": p["city"],
                  "target_date": p["target_date"], "fallback_used": p["fallback_used"],
                  "fallback_reason": p["fallback_reason"], "inputs_as_of": p["inputs_as_of"],
                  "feature_row": log.get("feature_row"), "persistence_c": log.get("persistence_c"),
                  "arm": log.get("arm")}
        out.append({**common, "model_version": str(p["model_version"]), "role": "served",
                    "prediction": p["forecast_max_c"]})
        out += [{**common, "model_version": str(s["model_version"]), "role": "shadow", "prediction": s["prediction"]}
                for s in log.get("shadows") or []]
    return out


def log_rows(values: list[bytes]) -> pd.DataFrame:
    """The predictions rows in the raw Kafka values, one per model per answered city, each key kept once."""
    df = pd.DataFrame([r for v in values for r in _rows(decode(v))], columns=COLUMNS)
    for c in ("target_date", "inputs_as_of", "feature_row"):
        df[c] = pd.to_datetime(df[c])
    df["prediction"] = pd.to_numeric(df["prediction"]).astype("float64")
    df["persistence_c"] = pd.to_numeric(df["persistence_c"]).astype("float64")
    return df.drop_duplicates(KEY).reset_index(drop=True)


def _predictions_group(fs):
    from hsfs.feature import Feature

    name, version = platform.PREDICTIONS
    fg = fs.get_or_create_feature_group(
        name=name, version=version, primary_key=KEY, event_time="request_time", online_enabled=False,
        features=[Feature(n, t) for n, t in SCHEMA.items()],
        description="Every model's prediction per answered request, from the predictor's inference log.")
    # an existing group keeps the schema it was created with; columns added since (arm) are appended, None in old rows
    have = {f.name for f in fg.features}
    missing = [Feature(n, t) for n, t in SCHEMA.items() if n not in have]
    if missing:
        fg.append_features(missing)
    return fg


def run_log() -> int:
    """Read the topic from where the last run stopped, write the rows, then commit the offsets: a run that fails
    before the write is read again by the next one. Returns the number of rows written."""
    from confluent_kafka import Consumer

    project = platform.login()
    deployment = project.get_model_serving().get_deployment(platform.DEPLOYMENT)
    topic = deployment.predictor.inference_logger.kafka_topic.name
    config = project.get_kafka_api().get_default_config()
    config.update({"group.id": CONSUMER_GROUP, "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer = Consumer(config)
    started = time.time()
    consumer.subscribe([topic], on_assign=lambda _, parts: print(
        f"assigned {len(parts)} partitions of {topic} after {time.time() - started:.0f}s"))
    values, quiet_since = [], time.time()
    try:
        while time.time() - quiet_since < IDLE_SECONDS:
            msg = consumer.poll(5)
            if msg is None:
                continue
            if msg.error():
                raise RuntimeError(f"Reading {topic}: {msg.error()}")
            values.append(msg.value())
            quiet_since = time.time()
        print(f"{len(values)} messages read")
        rows = log_rows(values)
        if len(rows):
            _predictions_group(project.get_feature_store()).insert(rows, wait=True)
        if values:
            consumer.commit(asynchronous=False)
        return len(rows)
    finally:
        consumer.close()
