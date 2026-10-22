"""The training pipeline's side of Hopsworks: training data from the feature view, and models in the registry.

Each model's state lives in its lifecycle_status tag, a JSON string (the cluster's tag schema is a plain string):
{"status": "candidate" | "challenger" | "rejected" | "champion", "window": {...}, "lineage": {...}, "gate": {...}}.
Promotion is by hand, by setting the status to champion; a tag holding just the word champion counts too.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from src import platform
from src.training_pipeline.window import TrainingWindow

STATUS_TAG = "lifecycle_status"
MODEL_FILE = "model.joblib"


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    training_dataset_version: int


def feature_view(fs):
    return fs.get_feature_view(*platform.FEATURE_VIEW)


def newest_labelled(fs, label: str) -> dict[str, date]:
    """The newest target date with a label in weather_features per city, as {city: date}. Rows for target dates
    that have no weather yet (served placeholders) have no label and are left out."""
    df = fs.get_feature_group(*platform.WEATHER_FEATURES).select(["city", "target_date", label]).read()
    df = df[df[label].notna()]
    return pd.to_datetime(df["target_date"]).dt.date.groupby(df["city"]).max().to_dict()


def _keep(X: pd.DataFrame, y: pd.DataFrame, first: date, last: date, label: str):
    """Rows with a target date from first to last (both included) and a label."""
    days = pd.to_datetime(X["target_date"]).dt.date
    keep = ((days >= first) & (days <= last) & y[label].notna()).to_numpy()
    return X[keep].reset_index(drop=True), y[label][keep].reset_index(drop=True)


def read_split(fv, window: TrainingWindow, label: str) -> Split:
    """Training and holdout rows, read in memory as a new training dataset version of the feature view.

    The bounds asked of Hopsworks are one day wider than the window, so the window holds whether Hopsworks
    includes the end date or not; the rows are then cut to the window exactly."""
    X_tr, X_ho, y_tr, y_ho = fv.train_test_split(
        train_start=window.train_start, train_end=window.holdout_start,
        test_start=window.holdout_start, test_end=window.holdout_end + timedelta(days=1),
        description=f"Training to {window.cutoff}, holdout {window.holdout_start} to {window.holdout_end}")
    version = fv.get_last_accessed_training_dataset()
    return Split(*_keep(X_tr, y_tr, window.train_start, window.cutoff, label),
                 *_keep(X_ho, y_ho, window.holdout_start, window.holdout_end, label), version)


def read_holdout(fv, training_dataset_version: int, window: TrainingWindow, label: str):
    """The holdout rows of a training dataset version, cut to the window exactly as read_split did."""
    _, X_ho, _, y_ho = fv.get_train_test_split(training_dataset_version)
    return _keep(X_ho, y_ho, window.holdout_start, window.holdout_end, label)


def register(project, fv, model, split: Split, window: TrainingWindow, lineage: dict) -> int:
    """Save the fitted pipeline as a new version of the model, linked to the feature view and the training dataset
    version, with the status candidate. Returns the version."""
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(model, Path(tmp) / MODEL_FILE)
        hs_model = project.get_model_registry().python.create_model(
            name=platform.MODEL, feature_view=fv, training_dataset_version=split.training_dataset_version,
            description=f"LightGBM trained to {window.cutoff} on {len(split.X_train)} rows, by the training pipeline")
        hs_model.save(tmp)
    record = {"status": "candidate", "window": {k: v.isoformat() for k, v in asdict(window).items()},
              "lineage": lineage}
    hs_model.add_tag(STATUS_TAG, json.dumps(record))
    return hs_model.version


def record(model) -> dict:
    """A model's lifecycle_status tag as a dict; {} when it has none."""
    value = (model.get_tags() or {}).get(STATUS_TAG)
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        parsed = value
    return parsed if isinstance(parsed, dict) else {"status": str(parsed).strip()}


def window_of(model) -> TrainingWindow:
    w = record(model).get("window")
    if not w:
        raise ValueError(f"Model version {model.version} has no training window in its {STATUS_TAG} tag.")
    return TrainingWindow(**{k: date.fromisoformat(v) for k, v in w.items()})


def set_status(model, status: str, gate: dict):
    model.add_tag(STATUS_TAG, json.dumps({**record(model), "status": status, "gate": gate}))


def champion(mr):
    """The model tagged champion, or None before there is one. Two champions is an error to fix by hand."""
    found = [m for m in mr.get_models(platform.MODEL) or [] if record(m).get("status") == "champion"]
    if len(found) > 1:
        raise ValueError(f"Versions {', '.join(str(m.version) for m in found)} are all tagged champion; keep one.")
    return found[0] if found else None


def load(model):
    """The fitted pipeline of a registered model."""
    return joblib.load(Path(model.download()) / MODEL_FILE)
