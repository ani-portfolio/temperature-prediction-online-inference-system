"""Which registered models serve, read from their lifecycle_status tags. Shared by the training and inference pipelines.

Each model's state lives in its lifecycle_status tag, a JSON string (the cluster's tag schema is a plain string):
{"status": "candidate" | "challenger" | "rejected" | "champion" | "retiring", ...}. Promotion is by hand: tag the new
model champion and the old one {"status": "retiring", "since": "<date>"}. A tag holding just a word counts too.
The predictor reads the tags once at startup, so a change takes effect when the deployment restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import joblib

from src import platform

STATUS_TAG = "lifecycle_status"
MODEL_FILE = "model.joblib"
RETIRING_DAYS = 28   # a former champion predicts in shadow this long after its replacement is promoted


@dataclass
class ServingSet:
    champion: object | None                         # None before the first promotion: the predictor serves persistence
    shadows: list = field(default_factory=list)     # predict next to the champion, never served


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


def champion(mr):
    """The model tagged champion, or None before there is one. Two champions is an error to fix by hand."""
    found = [m for m in mr.get_models(platform.MODEL) or [] if record(m).get("status") == "champion"]
    if len(found) > 1:
        raise ValueError(f"Versions {', '.join(str(m.version) for m in found)} are all tagged champion; keep one.")
    return found[0] if found else None


def _shadows(model, today: date) -> bool:
    tag = record(model)
    if tag.get("status") == "challenger":
        return True
    if tag.get("status") != "retiring":
        return False
    if not tag.get("since"):
        raise ValueError(f"Version {model.version} is tagged retiring with no since date; add the promotion date.")
    return today - date.fromisoformat(tag["since"]) < timedelta(days=RETIRING_DAYS)


def serving_models(mr, today: date) -> ServingSet:
    """The champion, and in shadow every challenger and every model retiring for less than RETIRING_DAYS, by version."""
    models = sorted(mr.get_models(platform.MODEL) or [], key=lambda m: m.version)
    return ServingSet(champion(mr), [m for m in models if _shadows(m, today)])


def load(model):
    """The fitted pipeline of a registered model."""
    return joblib.load(Path(model.download()) / MODEL_FILE)
