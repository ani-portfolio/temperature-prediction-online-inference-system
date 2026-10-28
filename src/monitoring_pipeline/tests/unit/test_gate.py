"""The live gate: a shadow model against the champion on the city-days both scored, after both training cutoffs."""

from datetime import date

import numpy as np
import pandas as pd

from src.monitoring_pipeline.gate import live_gate

AS_OF = date(2026, 12, 1)
CUTOFFS = {"3": date(2026, 9, 1), "4": date(2026, 10, 1)}


def scored(version, errors, first="2026-10-02", cities=("London", "Tokyo")):
    rows = []
    for i, e in enumerate(errors):
        for c in cities:
            rows.append({"city": c, "target_date": pd.Timestamp(first) + pd.Timedelta(days=i),
                         "model_version": version, "prediction": 10.0 + e, "actual": 10.0})
    return pd.DataFrame(rows)


def test_a_clearly_better_shadow_is_significant():
    rng = np.random.default_rng(0)
    champ = scored("3", 2.0 + rng.normal(0, 0.3, 40))
    shadow = scored("4", 1.0 + rng.normal(0, 0.3, 40))
    r = live_gate(pd.concat([champ, shadow]), "3", "4", CUTOFFS, AS_OF)
    assert r["p_one_sided"] < 0.05
    assert r["model_mae"] < r["champion_mae"]
    assert (r["days"], r["rows"], r["model_version"], r["champion_version"]) == (40, 80, "4", "3")


def test_only_days_both_models_scored_after_both_cutoffs_count():
    champ = scored("3", [1.0] * 10, first="2026-09-25")      # 7 days before the shadow's cutoff
    shadow = scored("4", [1.0, 2.0, 1.5], first="2026-10-02", cities=("London",))
    r = live_gate(pd.concat([champ, shadow]), "3", "4", CUTOFFS, AS_OF)
    assert (r["days"], r["rows"]) == (3, 3)
    assert r["first_day"] == "2026-10-02"


def test_no_shared_days_is_no_evidence():
    r = live_gate(scored("3", [1.0] * 5), "3", "4", CUTOFFS, AS_OF)
    assert (r["days"], r["p_one_sided"], r["model_mae"]) == (0, 1.0, None)


def test_identical_forecasts_are_no_evidence():
    r = live_gate(pd.concat([scored("3", [1.0, 2.0, 3.0]), scored("4", [1.0, 2.0, 3.0])]), "3", "4", CUTOFFS, AS_OF)
    assert (r["p_one_sided"], r["p_two_sided"]) == (1.0, 1.0)


def test_fewer_than_14_shared_days_is_no_evidence_yet():
    r = live_gate(pd.concat([scored("3", [3.0] * 13), scored("4", [0.5, 0.6] * 6 + [0.5])]), "3", "4", CUTOFFS, AS_OF)
    assert (r["days"], r["p_one_sided"]) == (13, 1.0)
    assert r["model_mae"] < r["champion_mae"]
