"""Which registered models serve, read from their lifecycle_status tags. Shared by the training, inference, and
monitoring pipelines.

Each model's state lives in its lifecycle_status tag, a JSON string (the cluster's tag schema is a plain string):
{"status": "candidate" | "challenger" | "rejected" | "canary" | "ab" | "champion" | "retiring", ...}. Promotion is
by hand: tag the new model champion and the old one {"status": "retiring", "since": "<date>"}. A tag holding just a
word counts too. The predictor reads the tags once at startup, so a change takes effect when the deployment restarts.

A rollout runs challenger (shadow) -> canary -> ab -> champion. A canary or ab tag also holds the model's share of
city-days and a salt, {"status": "canary", "traffic": 0.1, "salt": "<rollout id>"}; arm() picks each city-day's model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from src import platform

STATUS_TAG = "lifecycle_status"
MODEL_FILE = "model.joblib"
REFERENCE_FILE = "reference.parquet"   # out-of-fold predictions for the last training year, for prediction drift
RETIRING_DAYS = 28   # a former champion predicts in shadow this long after its replacement is promoted
CANDIDATE_STATUSES = ("canary", "ab")


@dataclass
class ServingSet:
    champion: object | None                         # None before the first promotion: the predictor serves persistence
    shadows: list = field(default_factory=list)     # predict next to the champion, never served
    candidate: object | None = None                 # the model in canary or ab, served on its share of city-days
    traffic: float = 0.0                            # the candidate's share of city-days
    salt: str | None = None                         # reshuffles which city-days the candidate gets, per rollout


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


def _candidate(models) -> tuple[object | None, float, str | None]:
    """The model in canary or ab with its share and salt; (None, 0.0, None) when there is none. Two candidates, or a
    candidate without a share in (0, 1], a salt, or its training window cutoff, is an error to fix by hand."""
    found = [m for m in models if record(m).get("status") in CANDIDATE_STATUSES]
    if len(found) > 1:
        raise ValueError(f"Versions {', '.join(str(m.version) for m in found)} are all in canary or ab; keep one.")
    if not found:
        return None, 0.0, None
    model, tag = found[0], record(found[0])
    traffic = tag.get("traffic")
    if isinstance(traffic, bool) or not isinstance(traffic, (int, float)) or not 0 < traffic <= 1:
        raise ValueError(f"Version {model.version} is in {tag['status']} with traffic {traffic!r}; set a share in (0, 1].")
    if not tag.get("salt"):
        raise ValueError(f"Version {model.version} is in {tag['status']} with no salt; add one for the rollout.")
    if not (tag.get("window") or {}).get("cutoff"):   # the live gate starts after it; lost when a tag is retyped
        raise ValueError(f"Version {model.version} is in {tag['status']} with no window cutoff; keep the window "
                         f"from its challenger tag.")
    return model, float(traffic), str(tag["salt"])


def serving_models(mr, today: date) -> ServingSet:
    """The champion, the candidate in canary or ab with its share, and in shadow every challenger and every model
    retiring for less than RETIRING_DAYS, by version."""
    models = sorted(mr.get_models(platform.MODEL) or [], key=lambda m: m.version)
    return ServingSet(champion(mr), [m for m in models if _shadows(m, today)], *_candidate(models))


def arm(city: str, target_date: date, salt: str, traffic: float) -> str:
    """'candidate' or 'champion' for one city-day. sha256 of salt|city|date puts the city-day in a bucket in [0, 1),
    and buckets below traffic go to the candidate, so every request for a city-day gets the same model, and raising
    traffic only moves city-days from the champion to the candidate."""
    digest = hashlib.sha256(f"{salt}|{city}|{target_date.isoformat()}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "candidate" if bucket < traffic else "champion"


def load(model):
    """The fitted pipeline of a registered model."""
    return joblib.load(Path(model.download()) / MODEL_FILE)


def load_reference(model) -> pd.DataFrame | None:
    """The reference predictions saved with a model (city, target_date, prediction), or None for a model saved
    without them."""
    path = Path(model.download()) / REFERENCE_FILE
    return pd.read_parquet(path) if path.exists() else None


def set_live(model, live: dict):
    """Write the live gate result into the model's tag as its live block, keeping every other field."""
    model.add_tag(STATUS_TAG, json.dumps({**record(model), "live": live}))
