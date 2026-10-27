"""Make the Hopsworks project run one commit of this repo: python -m deploy.apply [--ref SHA] [--dry-run]

In order: check the clone out at the commit, install the runtime requirements into both environments (weather for
jobs, weather-inference for the predictor) when uv.lock changed, upload src/ as the zip the predictor imports, create
or update every job in the settings, create or update the deployment and restart it, and record what was deployed.
A dry run reads the project and prints what it would do. Jobs are never deleted. The deployment's model only links it
to the registry: the predictor serves the model tagged champion, so promotion stays by hand."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from deploy.settings import (
    CLONE_DIR,
    ENVIRONMENTS,
    REPO_URL,
    SRC_ZIP,
    Change,
    DeploymentState,
    JobState,
    changes,
    deployment_changes,
    desired_deployment,
    desired_state,
    load_settings,
    src_zip,
)
from src import platform

STATE = "Resources/deploy/state.json"


@dataclass
class DeployReport:
    ref: str
    dry_run: bool
    steps: list[str] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    deployment: Change | None = None
    problems: list[str] = field(default_factory=list)

    def __str__(self):
        lines = [f"{'Dry run of the deploy' if self.dry_run else 'Deploy'} of {self.ref}", *self.steps]
        lines += [f"job {c.job}: {c.action}{' (' + c.detail + ')' if c.detail else ''}" for c in self.changes]
        if self.deployment:
            c = self.deployment
            lines.append(f"deployment {c.job}: {c.action}{' (' + c.detail + ')' if c.detail else ''}")
        lines += [f"PROBLEM: {p}" for p in self.problems]
        return "\n".join(lines)


def runtime_requirements(root: Path = platform.ROOT) -> str:
    """The runtime dependency group pinned from uv.lock, as a requirements.txt for the Hopsworks environment."""
    return subprocess.run(
        ["uv", "export", "--only-group", "runtime", "--no-hashes", "--no-emit-project", "--no-header", "--frozen"],
        cwd=root, check=True, capture_output=True, text=True).stdout


def lock_hash(root: Path = platform.ROOT) -> str:
    return hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()


def _live_state(job) -> JobState:
    config = job.config or {}
    resources = config.get("resourceConfig") or {}
    schedule = job.job_schedule
    return JobState(config.get("appPath", ""), config.get("environmentName", ""), config.get("defaultArgs") or "",
                    int(resources.get("memory", 0)), int(resources.get("cores", 0)),
                    schedule.cron_expression if schedule and schedule.enabled else None)


def _read_state(project) -> dict:
    datasets = project.get_dataset_api()
    if not datasets.exists(STATE):
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(datasets.download(STATE, tmp, overwrite=True)).read_text())


def _upload_bytes(project, data: bytes, name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / name
        local.write_bytes(data)
        return project.get_dataset_api().upload(str(local), str(Path(STATE).parent), overwrite=True)


def _upload_text(project, text: str, name: str) -> str:
    return _upload_bytes(project, text.encode(), name)


def _sync_clone(project, ref, report):
    git = project.get_git_api()
    name = Path(CLONE_DIR).name
    repo = git.get_repo(name)
    if report.dry_run:
        at = repo.current_commit.hash if repo else None
        report.steps.append(f"clone: would {'check out' if repo else 'clone and check out'} {ref} (now at {at})")
        return
    if repo is None:
        git.clone(REPO_URL, str(Path(CLONE_DIR).parent), "GitHub", "main")
        repo = git.get_repo(name)
    repo.fetch("origin", "main")
    repo.checkout_commit(ref)
    report.steps.append(f"clone: checked out {ref}")


def _sync_environments(project, root, state, report):
    envs = project.get_environment_api()
    changed = state.get("lock_hash") != lock_hash(root)
    requirements = None
    for name, base in ENVIRONMENTS.items():
        env = envs.get_environment(name)
        stale = env is None or changed
        if report.dry_run:
            report.steps.append(f"environment {name}: would {'create' if env is None else 'keep'}"
                                f"{' and install the runtime requirements' if stale else ''}")
            continue
        if env is None:
            env = envs.create_environment(name, base_environment_name=base)
            report.steps.append(f"environment {name}: created from {base}")
        if stale:
            requirements = requirements or _upload_text(project, runtime_requirements(root), "requirements.txt")
            env.install_requirements(requirements)
            report.steps.append(f"environment {name}: installed the runtime requirements")


def _upload_src(project, root, report):
    if report.dry_run:
        report.steps.append(f"{SRC_ZIP}: would upload src/ for the predictor")
        return
    _upload_bytes(project, src_zip(root), Path(SRC_ZIP).name)
    report.steps.append(f"{SRC_ZIP}: uploaded")


def _sync_jobs(project, settings, report):
    jobs = project.get_job_api()
    report.changes = changes(settings, {job.name: _live_state(job) for job in jobs.get_jobs()})
    if report.dry_run:
        return
    by_name = {job.name: job for job in settings.jobs}
    for change in report.changes:
        if change.action not in ("create", "update"):
            continue
        job, want = by_name[change.job], desired_state(by_name[change.job])
        config = jobs.get_configuration("PYTHON")
        config.update(appPath=want.app_path, environmentName=want.environment, defaultArgs=want.args)
        config["resourceConfig"].update(memory=want.memory_mb, cores=want.cores)
        made = jobs.create_job(job.name, config)
        if job.schedule:
            made.schedule(job.schedule)
        elif made.job_schedule:
            made.unschedule()


def _live_deployment(deployment) -> DeploymentState:
    logger = deployment.inference_logger
    return DeploymentState(deployment.script_file or "", deployment.environment or "",
                           (logger.mode if logger else None) or "NONE")


def _sync_deployment(project, settings, report):
    """Create the deployment when it is missing, update the settings that differ, and restart it so it loads the
    new code and reads the registry tags again."""
    if settings.deployment is None:
        return
    from hsml.inference_logger import InferenceLogger

    conf = settings.deployment
    serving = project.get_model_serving()
    deployment = serving.get_deployment(conf.name)
    want = desired_deployment(conf)
    change = deployment_changes(conf.name, want, deployment and _live_deployment(deployment))
    report.deployment = change
    if change.action == "create":
        models = project.get_model_registry().get_models(conf.model) or []
        if not models:   # a new project before its bootstrap: expected, so not a problem
            report.deployment = Change("skip", conf.name, f"no version of {conf.model} is registered yet, see "
                                                          f"'Starting a new project' in docs/deployment.md")
            return
        if report.dry_run:
            return
        newest = max(models, key=lambda m: m.version)
        predictor = serving.create_predictor(newest, name=conf.name, script_file=want.script_file,
                                             environment=want.environment, inference_logger=InferenceLogger())
        deployment = serving.create_deployment(predictor, name=conf.name)
        deployment.save()
        deployment.start()
        report.steps.append(f"deployment {conf.name}: created, linked to {conf.model} v{newest.version}, started")
        return
    if report.dry_run:
        report.steps.append(f"deployment {conf.name}: would restart to load the new code")
        return
    if change.action == "update":
        deployment.script_file = want.script_file
        deployment.environment = want.environment
        deployment.inference_logger = InferenceLogger()
        deployment.save()
    deployment.stop()
    deployment.start()
    report.steps.append(f"deployment {conf.name}: restarted")


def apply(ref: str, dry_run: bool = False, root: Path = platform.ROOT) -> DeployReport:
    settings = load_settings(root)
    report = DeployReport(ref, dry_run)
    project = platform.login()
    state = _read_state(project)
    report.steps.append(f"deployed now: {state.get('sha')} at {state.get('deployed_at')}")
    _sync_clone(project, ref, report)
    _sync_environments(project, root, state, report)
    _upload_src(project, root, report)
    _sync_jobs(project, settings, report)
    _sync_deployment(project, settings, report)
    if not dry_run and not report.problems:
        record = {"sha": ref, "lock_hash": lock_hash(root), "deployed_at": datetime.now(UTC).isoformat()}
        _upload_text(project, json.dumps(record, indent=2), Path(STATE).name)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="the commit to deploy (default: HEAD)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ref = a.ref or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
    report = apply(ref, a.dry_run)
    print(report)
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())
