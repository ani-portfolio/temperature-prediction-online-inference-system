"""Label feedback: join the actual maximum temperature to every logged model prediction, once.

Fallback rows (model_version 'persistence') are left out: their prediction is persistence_c, so they say nothing about
a model. Repeat requests for a city on a day use the same feature row, so the first row per city, target date, and
model is kept. A prediction whose actual has not arrived yet is scored by a later run.
"""

from __future__ import annotations

import pandas as pd

from src import platform

TARGET = "temperature_2m_max"
FALLBACK_VERSION = "persistence"
KEY = ["city", "target_date", "model_version"]
# The Hopsworks type of every column, declared so a batch whose column is all None still inserts
SCHEMA = {"city": "string", "target_date": "timestamp", "model_version": "string", "role": "string",
          "prediction": "double", "actual": "double", "abs_error": "double", "persistence_c": "double",
          "persistence_abs_error": "double", "request_time": "timestamp", "scored_at": "timestamp"}
COLUMNS = list(SCHEMA)


def score(predictions: pd.DataFrame, actuals: pd.DataFrame, already_scored: pd.DataFrame,
          now: pd.Timestamp) -> pd.DataFrame:
    """New scored_predictions rows. predictions: rows of the predictions feature group. actuals: city, date, and
    the target from weather_daily. already_scored: the KEY columns of scored_predictions."""
    p = predictions[predictions["model_version"].astype(str) != FALLBACK_VERSION].copy()
    p["model_version"] = p["model_version"].astype(str)
    p["target_date"] = pd.to_datetime(p["target_date"])
    p = p.sort_values("request_time", kind="stable").drop_duplicates(KEY)

    a = actuals[["city", "date", TARGET]].rename(columns={"date": "target_date", TARGET: "actual"})
    a = a.assign(target_date=pd.to_datetime(a["target_date"])).dropna(subset=["actual"])
    out = p.merge(a, on=["city", "target_date"], how="inner")

    done = already_scored[KEY].assign(target_date=lambda d: pd.to_datetime(d["target_date"]),
                                      model_version=lambda d: d["model_version"].astype(str))
    out = out.merge(done.assign(_done=True), on=KEY, how="left")
    out = out[out["_done"].isna()]

    out["abs_error"] = (out["prediction"] - out["actual"]).abs()
    out["persistence_abs_error"] = (out["persistence_c"] - out["actual"]).abs()
    out["scored_at"] = now
    return out[COLUMNS].sort_values(KEY).reset_index(drop=True)


def scored_group(fs):
    from hsfs.feature import Feature

    name, version = platform.SCORED_PREDICTIONS
    return fs.get_or_create_feature_group(
        name=name, version=version, primary_key=KEY, event_time="target_date", online_enabled=False,
        features=[Feature(n, t) for n, t in SCHEMA.items()],
        description="Every model's logged prediction joined to the actual maximum, by the monitoring job.")
