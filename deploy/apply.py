"""Make the Hopsworks project run one commit of this repo: python -m deploy.apply [--ref SHA] [--dry-run]

In order: check the clone out at the commit, install the runtime requirements into the environment when uv.lock
changed, create or update every job in the settings, restart the predictor, and record what was deployed. A dry
run reads the project and prints what it would do. Jobs are never deleted, and the model a deployment serves is
never changed: promotion stays by hand."""

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
    BASE_ENVIRONMENT,
    CLONE_DIR,
    ENVIRONMENT,
    REPO_URL,
    Change,
    JobState,
    changes,
    desired_state,
    load_settings,
)
from src import platform

STATE = "Resources/deploy/state.json"


@dataclass
class DeployReport:
    ref: str
    dry_run: bool
    steps: list[str] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def __str__(self):
        lines = [f"{'Dry run of the deploy' if self.dry_run else 'Deploy'} of {self.ref}", *self.steps]
        lines += [f"job {c.job}: {c.action}{' (' + c.detail + ')' if c.detail else ''}" for c in self.changes]
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


def _upload_text(project, text: str, name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / name
        local.write_text(text)
        return project.get_dataset_api().upload(str(local), str(Path(STATE).parent), overwrite=True)


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


def _sync_environment(project, root, state, report):
    envs = project.get_environment_api()
    env = envs.get_environment(ENVIRONMENT)
    stale = env is None or state.get("lock_hash") != lock_hash(root)
    if report.dry_run:
        report.steps.append(f"environment {ENVIRONMENT}: would {'create' if env is None else 'keep'}"
                            f"{' and install the runtime requirements' if stale else ''}")
        return
    if env is None:
        env = envs.create_environment(ENVIRONMENT, base_environment_name=BASE_ENVIRONMENT)
        report.steps.append(f"environment {ENVIRONMENT}: created from {BASE_ENVIRONMENT}")
    if stale:
        env.install_requirements(_upload_text(project, runtime_requirements(root), "requirements.txt"))
        report.steps.append(f"environment {ENVIRONMENT}: installed the runtime requirements")


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


def _restart_predictor(project, settings, report):
    if settings.deployment is None:
        return
    name = settings.deployment.name
    deployment = project.get_model_serving().get_deployment(name)
    if deployment is None:
        report.problems.append(f"deployment {name} does not exist; create it with the inference pipeline first")
    elif report.dry_run:
        report.steps.append(f"deployment {name}: would restart to load the new code")
    else:
        deployment.stop()
        deployment.start()
        report.steps.append(f"deployment {name}: restarted")


def apply(ref: str, dry_run: bool = False, root: Path = platform.ROOT) -> DeployReport:
    settings = load_settings(root)
    report = DeployReport(ref, dry_run)
    project = platform.login()
    state = _read_state(project)
    report.steps.append(f"deployed now: {state.get('sha')} at {state.get('deployed_at')}")
    _sync_clone(project, ref, report)
    _sync_environment(project, root, state, report)
    _sync_jobs(project, settings, report)
    _restart_predictor(project, settings, report)
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
