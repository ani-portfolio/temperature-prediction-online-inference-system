"""The daily monitoring job: label feedback, then the live gate, then the checks, each written to its feature group.

The live gate and the performance check read the scores, so they are skipped when label feedback fails; drift and
the served share do not need labels and still run. Any failed step fails the job at the end, so the platform's job
failure alert reaches the owner. Nothing here retrains or retags a model's status: the owner reads the reports.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from src import platform
from src.monitoring_pipeline import checks, gate, scores
from src.registry import load_reference, record, serving_models, set_live


@dataclass
class DailyReport:
    as_of: date
    scored_rows: int = 0
    gate_reports: list[dict] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _read(fs, name_version, columns=None) -> pd.DataFrame | None:
    """A feature group's rows, or None when it does not exist yet."""
    try:
        fg = fs.get_feature_group(*name_version)
    except Exception:   # noqa: BLE001 - the client raises when the group does not exist yet
        return None
    if fg is None:
        return None
    return (fg.select(columns) if columns else fg.select_all()).read()


def _as_of_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(as_of=pd.to_datetime(df["as_of"]))


def _cutoff(model) -> date:
    return date.fromisoformat(record(model)["window"]["cutoff"])


def _step(report: DailyReport, name: str, fn):
    try:
        return fn()
    except Exception:   # noqa: BLE001 - a failed step is recorded, the others still run, and the job fails at the end
        report.errors.append(f"{name}: {traceback.format_exc(limit=3)}")
        return None


def run_daily(today: date | None = None) -> DailyReport:
    """One day's monitoring. as_of is yesterday: the newest target date with an actual and the newest day with
    every request logged. Raises at the end when any step failed."""
    now = datetime.now(UTC)
    today = today or now.date()
    as_of = today - timedelta(days=1)
    report = DailyReport(as_of)
    project = platform.login()
    fs, mr = project.get_feature_store(), project.get_model_registry()
    predictions = _read(fs, platform.PREDICTIONS)
    if predictions is None:
        predictions = pd.DataFrame(columns=["model_version", "role", "fallback_reason", "request_time", "city",
                                            "target_date", "prediction", "persistence_c"])

    def label_feedback():
        old = _read(fs, platform.SCORED_PREDICTIONS)
        old = old if old is not None else pd.DataFrame(columns=scores.COLUMNS)
        actuals = _read(fs, platform.WEATHER_DAILY, ["city", "date", scores.TARGET])
        new = scores.score(predictions, actuals, old, pd.Timestamp(now))
        if len(new):
            scores.scored_group(fs).insert(new, wait=True)
        report.scored_rows = len(new)
        return pd.concat([old, new], ignore_index=True)

    scored = _step(report, "label feedback", label_feedback)
    serving = serving_models(mr, today)
    champ = serving.champion

    def live_gate():
        rows = []
        for shadow in serving.shadows:
            cutoffs = {str(champ.version): _cutoff(champ), str(shadow.version): _cutoff(shadow)}
            r = gate.live_gate(scored, str(champ.version), str(shadow.version), cutoffs, as_of)
            r["status"] = record(shadow).get("status")
            rows.append(r)
        if rows:
            gate.reports_group(fs).insert(_as_of_time(pd.DataFrame(rows)), wait=True)
            for shadow, r in zip(serving.shadows, rows):   # the table first: it is the record, the tag a copy
                set_live(shadow, r)
        report.gate_reports = rows

    if scored is not None and champ is not None:
        _step(report, "live gate", live_gate)

    def daily_checks():
        rows = [pd.DataFrame([checks.served_share_row(predictions, as_of)])]
        if champ is None:
            rows.append(checks.skipped_rows(["input_psi", "prediction_psi", "performance"], as_of, "no_champion"))
        else:
            fv = fs.get_feature_view(*platform.FEATURE_VIEW)
            X_train = fv.get_train_test_split(champ.training_dataset_version)[0]
            inputs = [c for c in X_train.columns if c not in ("city", "target_date")]
            live_inputs = _read(fs, platform.WEATHER_FEATURES, ["city", "target_date", *inputs])
            live_preds = None
            if scored is not None:
                live_preds = scored[scored["model_version"].astype(str) == str(champ.version)]
            rows.append(checks.drift_rows(X_train[["city", "target_date", *inputs]], live_inputs,
                                          load_reference(champ), live_preds, as_of))
            if scored is not None:
                rows.append(pd.DataFrame([checks.performance_row(scored, str(champ.version), as_of)]))
            else:
                rows.append(checks.skipped_rows(["performance"], as_of, "label_feedback_failed"))
        out = pd.concat(rows, ignore_index=True)
        checks.reports_group(fs).insert(_as_of_time(out), wait=True)
        report.flagged = [f"{r.check} {r.subject}: {r.value}" for r in out.itertuples() if r.flagged]

    _step(report, "checks", daily_checks)
    if report.errors:
        raise RuntimeError("Monitoring steps failed:\n" + "\n".join(report.errors))
    return report
