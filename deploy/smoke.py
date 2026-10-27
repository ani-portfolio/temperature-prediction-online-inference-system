"""Check a deploy works in prod: python -m deploy.smoke

Runs the feature job (feature_pipeline_hourly, scheduled daily) once (an upsert of the newest days, so an extra run is
harmless) and asks the deployment for one city's forecast, checking the response contract from the system design.
Before a new project's bootstrap there is no deployment and no data, so it only says so and passes; see 'Starting a
new project' in docs/deployment.md. Exits 1 on any problem."""

from __future__ import annotations

import sys

from deploy.settings import load_settings
from src import platform
from src.inference_pipeline.forecast import CONTRACT

SMOKE_JOB = "feature_pipeline_hourly"
SMOKE_CITY = "Auckland"
NO_CHAMPION = "no_champion"   # the predictor serves persistence until a model is tagged champion


def contract_problems(response: dict, city: str) -> list[str]:
    """What breaks the serving contract in one prediction, or nothing. Persistence before there is a champion is a
    fallback the contract allows."""
    out = [f"{key} is missing" for key in CONTRACT if key not in response]
    if "city" in response and response["city"] != city:
        out.append(f"city is {response['city']!r}, not {city!r}")
    forecast = response.get("forecast_max_c")
    if not isinstance(forecast, (int, float)) or not -90 <= forecast <= 60:
        out.append(f"forecast_max_c {forecast!r} is not a temperature between -90 and 60 C")
    if response.get("fallback_used") and response.get("fallback_reason") != NO_CHAMPION:
        out.append(f"fallback_used is true ({response.get('fallback_reason')}) right after a fresh feature run")
    return out


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
    execution = project.get_job_api().get_job(SMOKE_JOB).run(await_termination=True)
    if execution.final_status != "SUCCEEDED":
        problems.append(f"job {SMOKE_JOB} ended {execution.final_status}")
    if deployment:
        response = deployment.predict(inputs=[{"city": SMOKE_CITY}])["predictions"][0]
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
