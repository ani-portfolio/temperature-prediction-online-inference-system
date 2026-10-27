"""The job settings in each pipeline's jobs/jobs.yaml, and the changes that make the Hopsworks project match them.
Pure: no network, so CI checks every settings file before anything reaches Hopsworks."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src import platform

REPO_URL = "https://github.com/ani-portfolio/temperature-prediction-online-inference-system.git"
CLONE_DIR = "Resources/temperature-prediction-online-inference-system"
ENVIRONMENT = "weather"
BASE_ENVIRONMENT = "pandas-training-pipeline"   # has scikit-learn; the runtime group adds lightgbm and the pins
SRC_ZIP = "Resources/deploy/src.zip"            # src/ at the deployed commit, for the predictor (no HopsFS in pods)
INFERENCE_ENVIRONMENT = "weather-inference"     # deployments need an inference image; weather is a training image
INFERENCE_BASE_ENVIRONMENT = "pandas-inference-pipeline"
ENVIRONMENTS = {ENVIRONMENT: BASE_ENVIRONMENT, INFERENCE_ENVIRONMENT: INFERENCE_BASE_ENVIRONMENT}

JOB_KEYS = {"script", "args", "schedule", "memory_mb", "cores"}
DEPLOYMENT_KEYS = {"name", "model", "script"}


@dataclass(frozen=True)
class Job:
    name: str
    script: str               # path from the repo root
    args: str = ""
    schedule: str | None = None
    memory_mb: int = 2048
    cores: int = 1


@dataclass(frozen=True)
class Deployment:
    name: str
    model: str
    script: str               # path from the repo root


@dataclass(frozen=True)
class Settings:
    jobs: tuple[Job, ...]
    deployment: Deployment | None = None


@dataclass(frozen=True)
class JobState:
    """What Hopsworks holds for a job, in the fields the settings decide."""
    app_path: str
    environment: str
    args: str
    memory_mb: int
    cores: int
    schedule: str | None


@dataclass(frozen=True)
class DeploymentState:
    """What Hopsworks holds for the deployment, in the fields the settings decide."""
    script_file: str
    environment: str
    logging: str              # the inference logging mode: ALL sends requests and responses to the log topic


@dataclass(frozen=True)
class Change:
    action: str               # create, update, same, unmanaged (in the clone but in no settings file), or skip
    job: str                  # the job's name, or the deployment's
    detail: str = ""


def _refuse_unknown(keys, allowed, where):
    unknown = set(keys) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown settings {sorted(unknown)}; allowed {sorted(allowed)}")


def _script(jobs_dir: Path, root: Path, script, where) -> str:
    if not script:
        raise ValueError(f"{where}: a script is required")
    path = jobs_dir / script
    if not path.is_file():
        raise ValueError(f"{where}: script {script} is not in {jobs_dir.relative_to(root)}")
    return path.relative_to(root).as_posix()


def _check_cron(cron, where):
    if cron is not None and len(str(cron).split()) not in (6, 7):
        raise ValueError(f"{where}: schedule {cron!r} is not a Quartz cron (seconds minutes hours day month weekday)")


def load_settings(root: Path) -> Settings:
    """Every pipeline's jobs/jobs.yaml under root, checked. A bad file raises ValueError naming the setting."""
    jobs, deployment = {}, None
    for path in sorted(root.glob("src/*/jobs/jobs.yaml")):
        where = path.relative_to(root).as_posix()
        data = yaml.safe_load(path.read_text()) or {}
        _refuse_unknown(data, {"jobs", "deployment"}, where)
        for name, conf in (data.get("jobs") or {}).items():
            conf = conf or {}
            _refuse_unknown(conf, JOB_KEYS, f"{where} {name}")
            if name in jobs:
                raise ValueError(f"{where}: job {name} is already set in another pipeline")
            _check_cron(conf.get("schedule"), f"{where} {name}")
            jobs[name] = Job(name, _script(path.parent, root, conf.get("script"), f"{where} {name}"),
                             str(conf.get("args") or ""), conf.get("schedule"),
                             int(conf.get("memory_mb", 2048)), int(conf.get("cores", 1)))
        if data.get("deployment"):
            conf = data["deployment"]
            _refuse_unknown(conf, DEPLOYMENT_KEYS, f"{where} deployment")
            if deployment:
                raise ValueError(f"{where}: only one pipeline may set the deployment")
            deployment = Deployment(conf["name"], conf["model"],
                                    _script(path.parent, root, conf.get("script"), f"{where} deployment"))
    return Settings(tuple(jobs.values()), deployment)


def app_path(script: str) -> str:
    """A script in the clone, as Hopsworks stores a job's appPath."""
    return f"hdfs:///Projects/{platform.PROJECT}/{CLONE_DIR}/{script}"


def desired_state(job: Job) -> JobState:
    return JobState(app_path(job.script), ENVIRONMENT, job.args, job.memory_mb, job.cores, job.schedule)


def desired_deployment(deployment: Deployment) -> DeploymentState:
    return DeploymentState(f"/Projects/{platform.PROJECT}/{CLONE_DIR}/{deployment.script}", INFERENCE_ENVIRONMENT,
                           "ALL")


def _diff(have, want) -> str:
    return "; ".join(f"{f.name}: {getattr(have, f.name)!r} -> {getattr(want, f.name)!r}"
                     for f in fields(want) if getattr(have, f.name) != getattr(want, f.name))


def deployment_changes(name: str, want: DeploymentState, have: DeploymentState | None) -> Change:
    """What the deployment needs: create it, update what differs, or nothing (apply restarts it either way)."""
    if have is None:
        return Change("create", name)
    return Change("same", name) if have == want else Change("update", name, _diff(have, want))


def src_zip(root: Path) -> bytes:
    """src/ as a zip the predictor imports from: every .py file, without the pipelines' tests."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((root / "src").rglob("*.py")):
            rel = path.relative_to(root)
            if "tests" not in rel.parts and "__pycache__" not in rel.parts:
                z.write(path, rel.as_posix())
    return out.getvalue()


def changes(settings: Settings, live: dict[str, JobState]) -> list[Change]:
    """What each job in the settings needs, then the jobs that run from the clone but are in no settings file.
    Jobs Hopsworks made itself (statistics and the like) are left out."""
    out = []
    for job in settings.jobs:
        want, have = desired_state(job), live.get(job.name)
        if have is None:
            out.append(Change("create", job.name))
        elif have == want:
            out.append(Change("same", job.name))
        else:
            out.append(Change("update", job.name, _diff(have, want)))
    names = {job.name for job in settings.jobs}
    clone = app_path("")
    out += [Change("unmanaged", name) for name, state in sorted(live.items())
            if name not in names and state.app_path.startswith(clone)]
    return out
