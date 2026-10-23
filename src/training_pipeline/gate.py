"""The entry test of the champion/challenger gate: does a candidate beat the reference on the holdout?

The reference is the model tagged champion in the registry, or persistence (the last complete day's maximum) before
there is a champion. Both forecasts are scored on the same holdout rows, and src.compare tests the daily mean loss
difference, which allows for the correlation between cities on a day and between consecutive days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import platform
from src.compare import daily_loss_difference, diebold_mariano

TARGET = "temperature_2m_max"
PERSISTENCE = "temperature_2m_max_lag_2d"   # the last complete day's maximum: the reference before a champion


@dataclass
class GateReport:
    model_version: int
    reference: str          # 'champion vN' or 'persistence'
    candidate_mae: float
    reference_mae: float
    dm: dict
    status: str             # 'challenger' or 'rejected'

    @property
    def next_step(self) -> str:
        """What the owner does next: the predictor reads the tags only at startup."""
        if self.status != "challenger":
            return ""
        return f"Restart the deployment {platform.DEPLOYMENT} to shadow v{self.model_version}."


def entry_test(holdout: pd.DataFrame, candidate, reference, alpha: float = 0.05,
               target: str = TARGET) -> tuple[str, dict]:
    """'challenger' when a one sided Diebold-Mariano test at alpha says the candidate's MAE is lower than the
    reference's on the holdout, else 'rejected'. holdout holds city, target_date, and the target, one row per
    prediction. The result also holds both MAEs and the two sided p value, which tells a significantly worse
    candidate from one that is only not better."""
    candidate, reference = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if not len(holdout) == len(candidate) == len(reference):
        raise ValueError(f"The holdout has {len(holdout)} rows, but there are {len(candidate)} candidate and "
                         f"{len(reference)} reference predictions.")
    rows = pd.DataFrame({"row_id": holdout["city"].astype(str) + "|" + holdout["target_date"].astype(str),
                         "date": holdout["target_date"].to_numpy(), target: holdout[target].to_numpy()})
    d = daily_loss_difference(rows.assign(prediction=candidate), rows.assign(prediction=reference), target)
    if np.allclose(d, 0):   # the same forecasts: no evidence either way, and the test's variance would be 0
        dm = {"mean": 0.0, "se": 0.0, "stat": 0.0, "p_one_sided": 1.0, "p_two_sided": 1.0, "days": len(d)}
    else:
        dm = diebold_mariano(d)
    y = rows[target].to_numpy()
    dm |= {"candidate_mae": float(np.abs(candidate - y).mean()), "reference_mae": float(np.abs(reference - y).mean()),
           "alpha": alpha}
    return ("challenger" if dm["p_one_sided"] < alpha else "rejected"), dm


def run_gate(model_version: int) -> GateReport:
    """The gate job: score a registered candidate and its reference on the candidate's holdout, and tag the
    candidate challenger or rejected. The reference is the champion, or persistence before there is one."""
    from src.features import load_config
    from src.training_pipeline import registry

    label = load_config(root=platform.ROOT)["target"]
    project = platform.login()
    fv = registry.feature_view(project.get_feature_store())
    mr = project.get_model_registry()
    candidate = mr.get_model(platform.MODEL, model_version)
    if registry.record(candidate).get("status") == "champion":
        raise ValueError(f"Version {model_version} is the champion; the gate only judges candidates.")
    window = registry.window_of(candidate)
    X, y = registry.read_holdout(fv, candidate.training_dataset_version, window, label)
    predictions = registry.load(candidate).predict(X)

    champ = registry.champion(mr)
    if champ is None:
        reference, name = X[PERSISTENCE].to_numpy(dtype=float), "persistence"
        if np.isnan(reference).any():
            raise ValueError(f"{int(np.isnan(reference).sum())} holdout rows have no {PERSISTENCE}.")
    else:
        if registry.window_of(champ).cutoff >= window.holdout_start:
            raise ValueError(f"Champion v{champ.version} was trained on target dates in this holdout "
                             f"(from {window.holdout_start}); the comparison would favour it.")
        reference, name = registry.load(champ).predict(X), f"champion v{champ.version}"

    status, dm = entry_test(X[["city", "target_date"]].assign(**{label: y}), predictions, reference, target=label)
    registry.set_status(candidate, status, {"reference": name, **dm})
    return GateReport(model_version, name, dm["candidate_mae"], dm["reference_mae"], dm, status)
