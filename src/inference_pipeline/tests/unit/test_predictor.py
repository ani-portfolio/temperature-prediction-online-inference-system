"""The predictor shell: loads the serving models once, looks up each city's rows in one call, and answers per city."""

import ast
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from deploy.settings import SRC_ZIP
from src.inference_pipeline import predictor

NOON = datetime(2026, 10, 23, 23, 0, tzinfo=UTC)        # 12:00 on 2026-10-24 in Auckland: tomorrow is the 25th
PERSISTENCE = "temperature_2m_max_lag_2d"


class Model:
    def __init__(self, version, status):
        self.version, self.status = version, status

    def get_tags(self):
        return {"lifecycle_status": self.status}


class Registry:
    models = (Model(3, "champion"), Model(4, "challenger"))

    def get_models(self, name):
        return list(self.models)


class FeatureView:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def init_serving(self):
        self.calls.append("init")

    def get_feature_vectors(self, serving_keys, return_type, allow_missing):
        self.calls.append(serving_keys)
        assert return_type == "pandas" and allow_missing
        return pd.DataFrame([{**k, PERSISTENCE: self.rows.get((k["city"], k["target_date"].date()))}
                             for k in serving_keys])


class FeatureStore:
    def __init__(self, fv):
        self.fv = fv

    def get_feature_view(self, name, version):
        return self.fv


class Project:
    def __init__(self, fv):
        self.fv = fv

    def get_model_registry(self):
        return Registry()

    def get_feature_store(self):
        return FeatureStore(self.fv)


class Fixed:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value]


def server(rows):
    fv = FeatureView(rows)
    loaded = {3: Fixed(18.4), 4: Fixed(19.0)}
    return predictor.Server(Project(fv), today=date(2026, 10, 23), load=lambda m: loaded[m.version]), fv


def test_loads_the_champion_and_shadows_and_starts_serving():
    s, fv = server({})
    assert s.models.champion.version == 3
    assert [m.version for m in s.models.shadows] == [4]
    assert fv.calls == ["init"]


def test_loads_the_candidate_with_its_share(monkeypatch):
    monkeypatch.setattr(Registry, "models", (*Registry.models,
                                             Model(9, '{"status": "canary", "traffic": 0.1, "salt": "v9", '
                                                       '"window": {"cutoff": "2026-09-30"}}')))
    fv = FeatureView({})
    s = predictor.Server(Project(fv), today=date(2026, 10, 23), load=lambda m: Fixed(float(m.version)))
    assert s.models.candidate.version == 9
    assert s.models.candidate.pipeline.value == 9.0
    assert (s.models.traffic, s.models.salt) == (0.1, "v9")
    assert [m.version for m in s.models.shadows] == [4]


def test_answers_each_city_with_one_lookup():
    s, fv = server({("Auckland", date(2026, 10, 25)): 17.0})
    [r] = s([{"city": "Auckland"}], now=NOON)
    assert (r["forecast_max_c"], r["model_version"]) == (18.4, 3)
    assert r["log"]["shadows"] == [{"model_version": 4, "prediction": 19.0}]
    assert len(fv.calls) == 2 and len(fv.calls[1]) == 5


def test_each_request_gets_its_own_id():
    s, _ = server({("Auckland", date(2026, 10, 25)): 17.0})
    a, b = s([{"city": "Auckland"}, {"city": "Auckland"}], now=NOON)
    assert a["log"]["request_id"] != b["log"]["request_id"]


def test_an_unknown_city_or_no_rows_is_an_error_for_that_city_only():
    s, _ = server({("Auckland", date(2026, 10, 25)): 17.0})
    bad, none, good = s([{"city": "Atlantis"}, {"city": "Lima"}, {"city": "Auckland"}], now=NOON)
    assert set(bad) == {"city", "error"} and bad["city"] == "Atlantis"
    assert set(none) == {"city", "error"} and none["city"] == "Lima"
    assert good["forecast_max_c"] == 18.4


def test_the_entry_script_downloads_the_zip_deploy_uploads():
    entry = Path(predictor.__file__).parent / "jobs" / "predictor.py"
    constants = {n.targets[0].id: n.value.value for n in ast.parse(entry.read_text()).body
                 if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
    assert constants["SRC_ZIP"] == SRC_ZIP
