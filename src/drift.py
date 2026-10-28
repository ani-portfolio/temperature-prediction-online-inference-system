"""Population stability index (PSI): how far a live distribution has moved from a reference one.

The same formula as the offline evaluation (ten quantile bins from the reference, shares clipped at 1e-4), so the
0.25 threshold the evaluation used means the same thing in monitoring. Missing values are left out of both sides.
"""

import numpy as np


def psi(reference, live, bins=10) -> float:
    a, b = np.asarray(reference, dtype=float), np.asarray(live, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    edges = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(a, edges)[0] / len(a)
    q = np.histogram(b, edges)[0] / len(b)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(((p - q) * np.log(p / q)).sum())
