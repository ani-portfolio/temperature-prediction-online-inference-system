"""Check a deploy works in prod: python -m deploy.smoke

Runs the feature job (feature_pipeline_hourly, scheduled daily) once (an upsert of the newest days, so an extra run is
harmless), checks the monitoring job exists, and asks the deployment for one city's forecast, checking the response contract from the system design.
Before a new project's bootstrap there is no deployment and no data, so it only says so and passes; see 'Starting a
new project' in docs/deployment.md. Exits 1 on any problem."""

from __future__ import annotations

import sys
import time

from deploy.settings import load_settings
from src import platform
from src.inference_pipeline import forecast

SMOKE_JOB = "feature_pipeline_hourly"
PRESENT_JOBS = ("monitoring_pipeline_daily",)   # checked to exist, not run: a run reads and writes every report
SMOKE_CITY = "Auckland"
NO_CHAMPION = "no_champion"   # the predictor serves persistence until a model is tagged champion
RETRY_SECONDS = 60            # the gateway answers 503 until the restarted predictor is ready


def contract_problems(response: dict, city: str) -> list[str]:
    """What breaks the serving contract in one prediction, or nothing. Stricter than the contract itself: right after
    a fresh feature run the only fallback allowed is persistence before there is a champion."""
    out = forecast.contract_problems(response, city)
    if response.get("fallback_used") and response.get("fallback_reason") != NO_CHAMPION:
        out.append(f"fallback_used is true ({response.get('fallback_reason')}) right after a fresh feature run")
    return out


def _forecast(deployment, city: str) -> dict:
    """One city's prediction. Right after the restart the gateway can answer 503 (no healthy upstream) before the
    predictor is ready, so a 503 is retried once after RETRY_SECONDS; any other error, or a second 503, raises."""
    try:
        return deployment.predict(inputs=[{"city": city}])["predictions"][0]
    except Exception as e:   # the client raises its own error class, carrying the HTTP response
        if getattr(getattr(e, "response", None), "status_code", None) != 503:
            raise
        print(f"503 from the deployment, retrying in {RETRY_SECONDS}s")
    time.sleep(RETRY_SECONDS)
    return deployment.predict(inputs=[{"city": city}])["predictions"][0]


def smoke(project=None) -> list[str]:
    """The problems in prod, or nothing. project defaults to the logged in Hopsworks project."""
    settings = load_settings(platform.ROOT)
    project = project or platform.login()
    deployment = None
    if settings.deployment:
        deployment = project.get_model_serving().get_deployment(settings.deployment.name)
        if deployment is None:
            print(f"bootstrap pending: no deployment {settings.deployment.name} yet, see 'Starting a new project' "
                  f"in docs/deployment.md")
            return []
    problems = []
    for name in PRESENT_JOBS:
        try:
            found = project.get_job_api().get_job(name)
        except Exception:   # noqa: BLE001 - the client raises for a missing job; either way it is a problem
            found = None
        if found is None:
            problems.append(f"job {name} does not exist")
    execution = project.get_job_api().get_job(SMOKE_JOB).run(await_termination=True)
    if execution.final_status != "SUCCEEDED":
        problems.append(f"job {SMOKE_JOB} ended {execution.final_status}")
    if deployment:
        response = _forecast(deployment, SMOKE_CITY)
        problems += contract_problems(response, SMOKE_CITY)
        if response.get("fallback_reason") == NO_CHAMPION:
            print("serving persistence: no model tagged champion")
    return problems


def main():
    problems = smoke()
    print("\n".join(f"PROBLEM: {p}" for p in problems) or "smoke check passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
