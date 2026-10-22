"""Check a deploy works in prod: python -m deploy.smoke

Runs the hourly feature job once (an upsert of the newest days, so an extra run is harmless) and, once the
inference pipeline has a deployment, asks it for one city's forecast and checks the response contract from the
system design. Exits 1 on any problem."""

from __future__ import annotations

import sys

from deploy.settings import load_settings
from src import platform

SMOKE_JOB = "feature_pipeline_hourly"
SMOKE_CITY = "Auckland"
CONTRACT = ("city", "target_date", "forecast_max_c", "model_version", "fallback_used", "inputs_as_of")


def contract_problems(response: dict, city: str) -> list[str]:
    """What breaks the serving contract in one prediction, or nothing."""
    out = [f"{key} is missing" for key in CONTRACT if key not in response]
    if "city" in response and response["city"] != city:
        out.append(f"city is {response['city']!r}, not {city!r}")
    forecast = response.get("forecast_max_c")
    if not isinstance(forecast, (int, float)) or not -90 <= forecast <= 60:
        out.append(f"forecast_max_c {forecast!r} is not a temperature between -90 and 60 C")
    if response.get("fallback_used"):
        out.append("fallback_used is true right after a fresh feature run")
    return out


def smoke() -> list[str]:
    settings = load_settings(platform.ROOT)
    project = platform.login()
    problems = []
    execution = project.get_job_api().get_job(SMOKE_JOB).run(await_termination=True)
    if execution.final_status != "SUCCEEDED":
        problems.append(f"job {SMOKE_JOB} ended {execution.final_status}")
    if settings.deployment:
        deployment = project.get_model_serving().get_deployment(settings.deployment.name)
        response = deployment.predict(inputs=[{"city": SMOKE_CITY}])
        problems += contract_problems(response["predictions"][0], SMOKE_CITY)
    return problems


def main():
    problems = smoke()
    print("\n".join(f"PROBLEM: {p}" for p in problems) or "smoke check passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
