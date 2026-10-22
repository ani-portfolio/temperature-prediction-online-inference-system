"""The feature pipeline's side of the feature store: the two feature groups and the feature view.

weather_daily     raw daily rows, key city and date, offline only, validated on insert (strict)
weather_features  one row per city and target date, key city and target_date, online and offline
feature view      the model inputs, with the target as the label; serves training data and online vectors
"""

from __future__ import annotations

import pandas as pd

from src import platform
from src.feature_pipeline.validate import expectation_suite


def daily_group(fs):
    name, version = platform.WEATHER_DAILY
    fg = fs.get_or_create_feature_group(
        name=name, version=version, primary_key=["city", "date"], event_time="date", online_enabled=False,
        description="Daily weather per city and local date from the Open-Meteo archive, as ingested.")
    return fg


def ensure_suite(fg):
    """Attach the validation rules, rejecting a batch that breaks any of them."""
    if fg.get_expectation_suite() is None:
        fg.save_expectation_suite(expectation_suite(), validation_ingestion_policy="strict")


def features_group(fs, features_hash: str):
    name, version = platform.WEATHER_FEATURES
    return fs.get_or_create_feature_group(
        name=name, version=version, primary_key=["city", "target_date"], event_time="target_date",
        online_enabled=True,
        description=f"Model inputs and target per city and target date, built by src.features.build_features "
                    f"(features hash {features_hash}).")


def feature_view(fs, fg, inputs: list[str], label: str):
    name, version = platform.FEATURE_VIEW
    query = fg.select(["city", "target_date", *inputs, label])
    return fs.get_or_create_feature_view(name=name, version=version, query=query, labels=[label],
                                         description="Inputs of the next day maximum temperature forecast.")


def stored_daily(fg, since) -> pd.DataFrame:
    """The raw daily rows on or after `since`."""
    df = fg.filter(fg.date >= pd.Timestamp(since)).read()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df
