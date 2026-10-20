"""How fast a frozen model's accuracy decays, measured against a baseline that does not age.

Each CV fold (and the test set) scores a model trained once, up to the fold's training end, on the days after it.
The error ratio model MAE / baseline MAE per week since the training end separates ageing from the season: the
baseline (persistence) carries the season's difficulty but has nothing to go stale.
"""

import numpy as np
import pandas as pd


def _weekly(preds, target, train_end, by="fold"):
    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"])
    if by in df and isinstance(train_end, (dict, pd.Series)):
        end = df[by].map(pd.Series(train_end).pipe(pd.to_datetime))
    else:
        end = pd.Timestamp(train_end)
    df["week"] = ((df["date"] - end).dt.days - 1) // 7 + 1
    df["abs_err"] = (df["prediction"] - df[target]).abs()
    return df


def decay_curve(model_preds, baseline_preds, train_end, target="temperature_2m_max", by="fold"):
    """MAE of the model and the baseline per week since the training end, and their ratio.

    model_preds, baseline_preds: one model each, with date, prediction, the target, and the fold column `by`
    (CV) or none (test). train_end: {fold: last training date} for CV, or one date for the test set.
    Returns a DataFrame with week, n, model_mae, baseline_mae, and ratio (below 1 means the model is better).
    """
    m = _weekly(model_preds, target, train_end, by).groupby("week")["abs_err"].agg(["size", "mean"])
    b = _weekly(baseline_preds, target, train_end, by).groupby("week")["abs_err"].mean()
    out = pd.DataFrame({"n": m["size"], "model_mae": m["mean"], "baseline_mae": b}).dropna()
    out["ratio"] = out["model_mae"] / out["baseline_mae"]
    return out.reset_index()


def decay_slope(curve, days=30):
    """The ratio's change per `days` days, from a least squares line weighted by forecasts per week."""
    if len(curve) < 2:
        return float("nan")
    slope = np.polyfit(curve["week"], curve["ratio"], 1, w=np.sqrt(curve["n"]))[0]
    return float(slope * days / 7)
