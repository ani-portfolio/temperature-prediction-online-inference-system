"""Deploy this repo to the Hopsworks project: CI tooling, not a pipeline.

The interface: load_settings and changes (pure: the job settings and what Hopsworks needs to match them), apply
(the deploy itself, `python -m deploy.apply`), and smoke with contract_problems (the check after a deploy,
`python -m deploy.smoke`)."""

from deploy.apply import DeployReport, apply, runtime_requirements
from deploy.settings import Change, Deployment, Job, JobState, Settings, changes, desired_state, load_settings
from deploy.smoke import contract_problems, smoke

__all__ = ["Change", "DeployReport", "Deployment", "Job", "JobState", "Settings", "apply", "changes",
           "contract_problems", "desired_state", "load_settings", "runtime_requirements", "smoke"]
