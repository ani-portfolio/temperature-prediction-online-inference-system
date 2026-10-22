"""The train job: refit the champion's recipe on the newest data, register it as a candidate, and start the gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import yaml

from src import platform
from src.features import load_config
from src.training_pipeline import registry
from src.training_pipeline.model import build_model
from src.training_pipeline.window import training_window

HOLDOUT_DAYS = 28
GATE_JOB = "training_pipeline_gate"


@dataclass
class TrainingReport:
    model_version: int
    train_start: date
    cutoff: date
    holdout_start: date
    holdout_end: date
    train_rows: int
    training_dataset_version: int
    gate_started: bool


def _spec():
    return yaml.safe_load((platform.ROOT / "ml_spec.yaml").read_text())


def lineage(spec: dict) -> dict:
    """Where the model's recipe comes from: the experiment that tuned it and the feature code it expects."""
    champion = spec["training"]["champion"]
    return {"params": champion["params"], "tuning_run_id": champion["run_id"],
            "mlflow_run_id": spec["training"]["mlflow"]["run_id"], "feature_view": list(platform.FEATURE_VIEW),
            "features_hash": spec["features"]["code"]["features_hash"]}


def run_training(holdout_days: int = HOLDOUT_DAYS, start_gate: bool = True) -> TrainingReport:
    """Train on every labelled target date from the split's first training day up to the holdout, register the
    model as a candidate, and start the gate job on it (unless start_gate is False)."""
    spec = _spec()
    config = load_config(spec)
    label = config["target"]
    start = date.fromisoformat(spec["split"]["cv"]["folds"][0]["train_start"])

    project = platform.login()
    fs = project.get_feature_store()
    window = training_window(registry.newest_labelled(fs, label), start, holdout_days)
    fv = registry.feature_view(fs)
    split = registry.read_split(fv, window, label)
    model = build_model(spec["training"]["champion"]["params"], config).fit(split.X_train, split.y_train)
    version = registry.register(project, fv, model, split, window, lineage(spec))
    if start_gate:
        project.get_job_api().get_job(GATE_JOB).run(args=f"--version {version}", await_termination=False)
    return TrainingReport(version, window.train_start, window.cutoff, window.holdout_start, window.holdout_end,
                          len(split.X_train), split.training_dataset_version, start_gate)
