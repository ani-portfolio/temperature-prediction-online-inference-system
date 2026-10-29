"""The daily checks, one monitoring_reports row each: drift, live performance, and the model-served share; and the
canary guardrail, one canary_reports row per candidate in canary or ab.

Drift compares the last WINDOW_DAYS with the same calendar days in every training year, since the evaluation showed
the seasons alone move the distributions (PSI 0.31 on predictions). Input drift reads weather_features for every
city, so it does not depend on traffic; prediction drift keeps the reference to the cities seen live, so a change in
which cities are requested does not look like drift. A flag is for the owner to read: nothing is retrained here.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from src import platform
from src.compare import daily_loss_difference
from src.drift import psi
from src.monitoring_pipeline.gate import one_sided_better

WINDOW_DAYS = 28
PSI_ALERT = 0.25
ALPHA = 0.05
SERVED_SHARE_ALERT = 0.95
NOT_COUNTED = "awaiting_today"  # the gap after local midnight until the daily feature run, by design
FALLBACK_VERSION = "persistence"
EVALUATION_MONTHLY_MAE = (1.79, 2.44)
# The Hopsworks type of every column, declared so a batch whose column is all None (value, threshold, note) still
# inserts
SCHEMA = {"as_of": "timestamp", "check": "string", "subject": "string", "value": "double", "threshold": "double",
          "flagged": "boolean", "rows": "bigint", "note": "string"}
CHECK_COLUMNS = list(SCHEMA)
# On the test set the champion was never more than 12.1 C from persistence in 9,100 city-days, and the actual was
# more than 15 C from it on 0.14%: past this a forecast is broken (Kelvin, a sign flip), not a bold call
IMPLAUSIBLE_C = 15.0
CONTRACT_RANGE_C = (-90.0, 60.0)   # the serving contract's temperature range, from forecast.contract_problems
CANARY_SCHEMA = {"as_of": "timestamp", "model_version": "string", "status": "string", "traffic": "double",
                 "since": "string", "candidate_city_days": "bigint", "checked_city_days": "bigint",
                 "implausible": "bigint", "missing": "bigint", "breaches": "bigint", "threshold_c": "double",
                 "flagged": "boolean"}


def _row(as_of, check, subject, value=None, threshold=None, flagged=False, rows=0, note=None) -> dict:
    return {"as_of": as_of.isoformat(), "check": check, "subject": subject,
            "value": None if value is None else float(value), "threshold": threshold, "flagged": bool(flagged),
            "rows": int(rows), "note": note}


def _days(df):
    return pd.to_datetime(df["target_date"]).dt.date


def live_window(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Rows with a target date in the WINDOW_DAYS ending on as_of."""
    d = _days(df)
    return df[((d > as_of - timedelta(days=WINDOW_DAYS)) & (d <= as_of)).to_numpy()]


def same_calendar_days(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Rows whose target date falls on one of the window's calendar days (month and day), in any year."""
    window = {((as_of - timedelta(days=k)).month, (as_of - timedelta(days=k)).day) for k in range(WINDOW_DAYS)}
    t = pd.to_datetime(df["target_date"])
    return df[[(m, d) in window for m, d in zip(t.dt.month, t.dt.day)]]


def drift_rows(reference_inputs: pd.DataFrame, live_inputs: pd.DataFrame, reference_preds: pd.DataFrame | None,
               live_preds: pd.DataFrame | None, as_of: date) -> pd.DataFrame:
    """input_psi rows, one per input column (every column but city and target_date), and one prediction_psi row.
    reference_inputs: the champion's training rows. live_inputs: weather_features rows. reference_preds: the
    champion's reference predictions, or None. live_preds: the champion's scored predictions."""
    ref, live = same_calendar_days(reference_inputs, as_of), live_window(live_inputs, as_of)
    out = [_row(as_of, "input_psi", c, v := psi(ref[c], live[c]), PSI_ALERT, v > PSI_ALERT, live[c].notna().sum())
           for c in reference_inputs.columns.drop(["city", "target_date"])]
    if reference_preds is None:
        out.append(_row(as_of, "prediction_psi", "prediction", threshold=PSI_ALERT, note="no_reference"))
    else:
        lp = live_window(live_preds, as_of)
        rp = same_calendar_days(reference_preds[reference_preds["city"].isin(set(lp["city"]))], as_of)
        if lp.empty or rp.empty:
            out.append(_row(as_of, "prediction_psi", "prediction", threshold=PSI_ALERT, note="no_live_rows"))
        else:
            v = psi(rp["prediction"], lp["prediction"])
            out.append(_row(as_of, "prediction_psi", "prediction", v, PSI_ALERT, v > PSI_ALERT, len(lp)))
    return pd.DataFrame(out, columns=CHECK_COLUMNS)


def performance_row(scored: pd.DataFrame, served_version: str, as_of: date) -> dict:
    """The served model's live MAE over the last WINDOW_DAYS, flagged when a one sided Diebold-Mariano test cannot
    say it beats persistence on the same city-days. The note holds persistence's MAE, the p value, and the
    evaluation's monthly MAE range to compare with."""
    s = live_window(scored[(scored["model_version"].astype(str) == served_version) & (scored["role"] == "served")],
                    as_of)
    if s.empty:
        return _row(as_of, "performance", served_version, threshold=ALPHA, note="no_live_rows")
    ids = s["city"] + "|" + s["target_date"].astype(str)
    base = pd.DataFrame({"row_id": ids, "date": s["target_date"], "actual": s["actual"]})
    d = daily_loss_difference(base.assign(prediction=s["prediction"].to_numpy()),
                              base.assign(prediction=s["persistence_c"].to_numpy()), "actual")
    p = one_sided_better(d)["p_one_sided"]
    note = {"persistence_mae": float((s["persistence_c"] - s["actual"]).abs().mean()), "p_model_better": p,
            "days": len(d), "evaluation_monthly_mae": list(EVALUATION_MONTHLY_MAE)}
    return _row(as_of, "performance", served_version, (s["prediction"] - s["actual"]).abs().mean(), ALPHA,
                p >= ALPHA, len(s), json.dumps(note))


def served_share_row(predictions: pd.DataFrame, day: date) -> dict:
    """The share of the day's served answers (by request time, UTC) that came from a model, flagged under
    SERVED_SHARE_ALERT. Answers in the gap after local midnight (awaiting_today) are not counted."""
    p = predictions[(predictions["role"] == "served")
                    & (pd.to_datetime(predictions["request_time"], utc=True).dt.date == day)
                    & (predictions["fallback_reason"].fillna("") != NOT_COUNTED)]
    if p.empty:
        return _row(day, "served_share", "requests", threshold=SERVED_SHARE_ALERT, note="no_requests")
    share = (p["model_version"].astype(str) != FALLBACK_VERSION).mean()
    reasons = p["fallback_reason"].dropna().value_counts().to_dict()
    return _row(day, "served_share", "requests", share, SERVED_SHARE_ALERT, share < SERVED_SHARE_ALERT, len(p),
                json.dumps(reasons) if reasons else None)


def canary_row(predictions: pd.DataFrame, candidate_version: str, status: str, traffic: float, since: date | None,
               as_of: date) -> dict:
    """The canary guardrail for the candidate, over requests (by request time, UTC) from since, the day the current
    step began (every logged request when None), to as_of. Counted per city-day, however many requests it had:
    candidate_city_days were assigned to the candidate; checked_city_days had a candidate prediction or a failure;
    a breach is a candidate forecast more than IMPLAUSIBLE_C from persistence or outside the contract range, or a
    missing one (None in shadow, or a model_error on its own city-day, where the caller got persistence). Flagged
    on any breach."""
    day = pd.to_datetime(predictions["request_time"], utc=True).dt.date
    p = predictions[((day <= as_of) & (day >= since if since else True)).to_numpy()]
    p = p.assign(key=list(zip(p["city"], pd.to_datetime(p["target_date"]).dt.date)))
    own = p[p["model_version"].astype(str) == candidate_version]
    pred = pd.to_numeric(own["prediction"], errors="coerce")
    low, high = CONTRACT_RANGE_C
    far = ((pred - pd.to_numeric(own["persistence_c"], errors="coerce")).abs() > IMPLAUSIBLE_C) | ~pred.between(low, high)
    failed_served = p[(p["arm"] == "candidate") & (p["role"] == "served") & (p["fallback_reason"] == "model_error")]
    implausible = set(own.loc[pred.notna() & far, "key"])
    missing = set(own.loc[pred.isna(), "key"]) | set(failed_served["key"])
    breaches = implausible | missing
    return {"as_of": as_of.isoformat(), "model_version": candidate_version, "status": status, "traffic": float(traffic),
            "since": since and since.isoformat(), "candidate_city_days": p.loc[p["arm"] == "candidate", "key"].nunique(),
            "checked_city_days": len(set(own["key"]) | set(failed_served["key"])), "implausible": len(implausible),
            "missing": len(missing), "breaches": len(breaches), "threshold_c": IMPLAUSIBLE_C,
            "flagged": bool(breaches)}


def canary_group(fs):
    from hsfs.feature import Feature

    name, version = platform.CANARY_REPORTS
    return fs.get_or_create_feature_group(
        name=name, version=version, primary_key=["as_of", "model_version"], event_time="as_of", online_enabled=False,
        features=[Feature(n, t) for n, t in CANARY_SCHEMA.items()],
        description="The canary guardrail for the candidate in canary or ab: implausible or missing forecasts, by the monitoring job.")


def skipped_rows(checks: list[str], as_of: date, note: str) -> pd.DataFrame:
    """A row with no value per check that could not run, saying why (for example no_champion)."""
    return pd.DataFrame([_row(as_of, c, "all", note=note) for c in checks], columns=CHECK_COLUMNS)


def reports_group(fs):
    from hsfs.feature import Feature

    name, version = platform.MONITORING_REPORTS
    return fs.get_or_create_feature_group(
        name=name, version=version, primary_key=["as_of", "check", "subject"], event_time="as_of",
        online_enabled=False, features=[Feature(n, t) for n, t in SCHEMA.items()], description="Drift, live performance, and the model-served share, by the monitoring job.")
