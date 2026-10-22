"""Is one forecast better than another, beyond noise? Paired comparisons that respect correlated errors.

Rows on the same date share weather across cities, and consecutive days are correlated, so the rows are not
independent. Both tests work on the daily mean loss difference (one value per date), and allow for correlation
between days.
"""

import math

import numpy as np
import pandas as pd


def daily_loss_difference(a, b, target="temperature_2m_max"):
    """Mean absolute error of b minus a per date. a and b hold the same rows (row_id, date, prediction, target).
    Positive values mean a is better."""
    m = a[["row_id", "date", target, "prediction"]].merge(b[["row_id", "prediction"]], on="row_id",
                                                           suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("The two forecasts must cover the same rows.")
    m["d"] = (m["prediction_b"] - m[target]).abs() - (m["prediction_a"] - m[target]).abs()
    return m.groupby(pd.to_datetime(m["date"]))["d"].mean().sort_index()


def diebold_mariano(d, lag=7):
    """Diebold-Mariano test on a daily loss difference series d, with a Newey-West variance (Bartlett weights,
    `lag` days) and the Harvey small sample correction. Returns mean, standard error, statistic, and the one sided
    p value for 'a is better' (mean above 0) and the two sided p value, from Student's t with n - 1 df."""
    x = np.asarray(d, dtype=float)
    n = len(x)
    u = x - x.mean()
    var = u @ u / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    se = math.sqrt(var / n)
    stat = x.mean() / se
    stat *= math.sqrt((n + 1 - 2 * (lag + 1) + (lag + 1) * lag / n) / n)  # Harvey, Leybourne and Newbold (1997)
    from scipy import stats
    p_one = float(stats.t.sf(stat, df=n - 1))
    return {"mean": float(x.mean()), "se": se, "stat": float(stat), "p_one_sided": p_one,
            "p_two_sided": float(2 * min(p_one, 1 - p_one)), "days": n, "lag": lag}


def block_bootstrap(d, block=7, n=2000, seed=0):
    """Moving block bootstrap of the mean daily loss difference: resamples runs of `block` days, keeping the
    correlation inside each week. Returns the mean, its 95% interval, and the share of draws at or below 0."""
    x = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    starts = np.arange(len(x) - block + 1)
    k = math.ceil(len(x) / block)
    draws = np.array([np.concatenate([x[s:s + block] for s in rng.choice(starts, k)])[:len(x)].mean()
                      for _ in range(n)])
    return {"mean": float(x.mean()), "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)), "share_at_or_below_0": float((draws <= 0).mean())}
