"""The Hopsworks project and the names of what the pipelines share through it, in one place.

Every pipeline logs in here and refers to the feature groups, the feature view, and the model by these names and
versions, so they cannot drift apart. The API key comes from HOPSWORKS_API_KEY, or the project's .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = "online_inference_v2"
ROOT = Path(__file__).resolve().parents[1]

WEATHER_DAILY = ("weather_daily", 1)          # raw daily weather per city and local date
WEATHER_FEATURES = ("weather_features", 1)    # model inputs and target per city and target date
FEATURE_VIEW = ("temperature_max_forecast", 1)
MODEL = "temperature_max_forecast"
DEPLOYMENT = "temperatureforecast"            # KServe deployment names are letters and digits only
PREDICTIONS = ("predictions", 1)              # every model's prediction per answered request, from the log job
SCORED_PREDICTIONS = ("scored_predictions", 1)  # model predictions joined to the actuals, by the monitoring job
GATE_REPORTS = ("gate_reports", 1)            # each shadow model against the champion on live days, per day
MONITORING_REPORTS = ("monitoring_reports", 1)  # drift, live performance, and the model-served share, per day
CANARY_REPORTS = ("canary_reports", 1)        # the canary guardrail for the candidate in canary or ab, per day


def api_key() -> str | None:
    key = os.environ.get("HOPSWORKS_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "HOPSWORKS_API_KEY":
                key = value.strip().strip("'\"")
    return key or None


def login():
    """The Hopsworks project. With an API key (environment or .env), log in to PROJECT; without one, use the
    platform's own login, which works inside Hopsworks jobs and deployments."""
    import hopsworks

    key = api_key()
    return hopsworks.login(project=PROJECT, api_key_value=key) if key else hopsworks.login()
