"""deploy: the job settings in each pipeline's jobs/jobs.yaml, the changes that make Hopsworks match them, and the
requirements the Hopsworks environment installs."""

import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from deploy import Change, JobState, changes, contract_problems, desired_state, load_settings, runtime_requirements
from deploy.settings import (
    INFERENCE_ENVIRONMENT,
    Deployment,
    DeploymentState,
    deployment_changes,
    desired_deployment,
    src_zip,
)
from src import platform

ROOT = Path(__file__).resolve().parents[2]
HOURLY = """
jobs:
  feature_pipeline_hourly:
    script: hourly.py
    schedule: "0 15 * * * ?"
"""


def project(tmp_path, files):
    """A repo root with the given files, each path relative to the root."""
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def feature_project(tmp_path, yaml=HOURLY, **more):
    return project(tmp_path, {"src/feature_pipeline/jobs/jobs.yaml": yaml,
                              "src/feature_pipeline/jobs/hourly.py": "", **more})


def test_a_job_gets_its_script_path_from_the_repo_root_and_the_defaults(tmp_path):
    [job] = load_settings(feature_project(tmp_path)).jobs
    assert (job.name, job.script, job.schedule) == (
        "feature_pipeline_hourly", "src/feature_pipeline/jobs/hourly.py", "0 15 * * * ?")
    assert (job.args, job.memory_mb, job.cores) == ("", 2048, 1)


def test_a_job_without_a_schedule_runs_on_demand(tmp_path):
    root = feature_project(tmp_path, yaml="jobs:\n  feature_pipeline_hourly:\n    script: hourly.py\n")
    assert load_settings(root).jobs[0].schedule is None


@pytest.mark.parametrize("yaml, message", [
    (HOURLY + "    retries: 2\n", "retries"),
    ("jobs:\n  feature_pipeline_hourly:\n    script: missing.py\n", "missing.py"),
    ("jobs:\n  feature_pipeline_hourly:\n    script: hourly.py\n    schedule: '15 * * * *'\n", "cron"),
    ("jobs:\n  feature_pipeline_hourly:\n    args: ''\n", "script"),
    ("jobs: {}\nextra: 1\n", "extra"),
])
def test_a_bad_setting_is_refused_with_its_name(tmp_path, yaml, message):
    with pytest.raises(ValueError, match=message):
        load_settings(feature_project(tmp_path, yaml=yaml))


def test_two_pipelines_cannot_share_a_job_name(tmp_path):
    root = feature_project(tmp_path, **{"src/training_pipeline/jobs/jobs.yaml": HOURLY,
                                        "src/training_pipeline/jobs/hourly.py": ""})
    with pytest.raises(ValueError, match="feature_pipeline_hourly"):
        load_settings(root)


def test_the_deployment_is_read_from_the_inference_pipeline(tmp_path):
    root = feature_project(tmp_path, **{
        "src/inference_pipeline/jobs/jobs.yaml":
            "deployment:\n  name: forecast\n  model: temperature_max_forecast\n  script: predictor.py\n",
        "src/inference_pipeline/jobs/predictor.py": ""})
    deployment = load_settings(root).deployment
    assert (deployment.name, deployment.model, deployment.script) == (
        "forecast", "temperature_max_forecast", "src/inference_pipeline/jobs/predictor.py")


def test_the_project_settings_load():
    settings = load_settings(ROOT)
    names = [job.name for job in settings.jobs]
    assert {"feature_pipeline_hourly", "inference_pipeline_log"} <= set(names)
    assert (settings.deployment.name, settings.deployment.model) == (platform.DEPLOYMENT, platform.MODEL)


@pytest.fixture
def settings(tmp_path):
    return load_settings(feature_project(tmp_path))


def test_a_job_missing_from_hopsworks_is_created(settings):
    assert changes(settings, {}) == [Change("create", "feature_pipeline_hourly")]


def test_a_job_that_matches_needs_nothing(settings):
    live = {"feature_pipeline_hourly": desired_state(settings.jobs[0])}
    assert changes(settings, live) == [Change("same", "feature_pipeline_hourly")]


def test_a_job_that_differs_is_updated_with_what_differs(settings):
    live = {"feature_pipeline_hourly": replace(desired_state(settings.jobs[0]), schedule=None, memory_mb=1024)}
    [change] = changes(settings, live)
    assert (change.action, change.job) == ("update", "feature_pipeline_hourly")
    assert "schedule" in change.detail and "memory_mb" in change.detail


def test_a_job_from_the_clone_with_no_settings_is_reported_and_others_are_ignored(settings):
    ours = desired_state(settings.jobs[0])
    live = {"feature_pipeline_hourly": ours,
            "old_job": replace(ours, app_path=ours.app_path.replace("hourly.py", "old.py")),
            "weather_daily_1_compute_stats": JobState("local:///srv/hops/artifacts/hsfs_utils.py", "", "", 2048, 1,
                                                      None)}
    assert Change("unmanaged", "old_job") in changes(settings, live)
    assert all(c.job != "weather_daily_1_compute_stats" for c in changes(settings, live))


def test_the_runtime_requirements_hold_only_what_runs_on_hopsworks():
    names = {line.split("==")[0] for line in runtime_requirements(ROOT).splitlines()
             if line and not line.startswith(("#", " "))}
    assert {"pandas", "lightgbm", "scikit-learn", "pyyaml"} <= names
    assert not names & {"hopsworks", "catboost", "xgboost", "optuna", "mlflow-skinny", "ipykernel", "pytest", "ruff"}


GOOD = {"city": "Auckland", "target_date": "2026-10-23", "forecast_max_c": 17.4, "model_version": 1,
        "fallback_used": False, "fallback_reason": None, "inputs_as_of": "2026-10-21"}


def test_a_response_that_keeps_the_contract_passes():
    assert contract_problems(GOOD, "Auckland") == []


@pytest.mark.parametrize("change, message", [
    ({"forecast_max_c": None}, "forecast_max_c"),
    ({"forecast_max_c": 75.0}, "forecast_max_c"),
    ({"city": "Moscow"}, "city"),
    ({"fallback_used": True}, "fallback"),
])
def test_a_response_that_breaks_the_contract_names_the_field(change, message):
    assert any(message in p for p in contract_problems({**GOOD, **change}, "Auckland"))


def test_a_response_missing_a_field_names_it():
    response = {k: v for k, v in GOOD.items() if k != "inputs_as_of"}
    assert any("inputs_as_of" in p for p in contract_problems(response, "Auckland"))


def test_a_response_missing_the_fallback_reason_names_it():
    response = {k: v for k, v in GOOD.items() if k != "fallback_reason"}
    assert any("fallback_reason" in p for p in contract_problems(response, "Auckland"))


DEPLOYMENT = Deployment("temperatureforecast", "temperature_max_forecast", "src/inference_pipeline/jobs/predictor.py")


def test_the_deployment_runs_the_clone_script_in_the_inference_environment_with_logging():
    want = desired_deployment(DEPLOYMENT)
    assert want.script_file.endswith("/Resources/temperature-prediction-online-inference-system/"
                                     "src/inference_pipeline/jobs/predictor.py")
    assert want.script_file.startswith(f"/Projects/{platform.PROJECT}/")
    assert (want.environment, want.logging) == (INFERENCE_ENVIRONMENT, "ALL")


def test_a_missing_deployment_is_created():
    assert deployment_changes(DEPLOYMENT.name, desired_deployment(DEPLOYMENT), None) == Change("create", DEPLOYMENT.name)


def test_a_matching_deployment_is_only_restarted():
    want = desired_deployment(DEPLOYMENT)
    assert deployment_changes(DEPLOYMENT.name, want, want) == Change("same", DEPLOYMENT.name)


def test_a_deployment_that_differs_is_updated_with_what_differs():
    want = desired_deployment(DEPLOYMENT)
    have = DeploymentState("/Projects/x/old.py", "pandas-inference-pipeline", "NONE")
    change = deployment_changes(DEPLOYMENT.name, want, have)
    assert change.action == "update"
    assert all(f in change.detail for f in ("script_file", "environment", "logging"))


def test_the_src_zip_holds_the_code_without_tests_and_imports(tmp_path):
    path = tmp_path / "src.zip"
    path.write_bytes(src_zip(ROOT))
    names = zipfile.ZipFile(path).namelist()
    assert "src/__init__.py" in names and "src/inference_pipeline/forecast.py" in names
    assert not [n for n in names if "/tests/" in n or "__pycache__" in n or not n.endswith(".py")]
    code = "import sys; sys.path.insert(0, sys.argv[1]); import src.inference_pipeline.forecast as f; print(f.__file__)"
    out = subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=False, capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "src.zip" in out.stdout
