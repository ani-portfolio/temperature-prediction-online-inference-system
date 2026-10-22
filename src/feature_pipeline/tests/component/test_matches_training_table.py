"""The feature pipeline builds exactly the inputs the champion was trained on (no training and serving skew)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.feature_pipeline import feature_rows
from src.features import feature_columns, load_config

ROOT = Path(__file__).resolve().parents[4]
CITIES = ["Auckland", "Moscow", "Bogota"]   # southern, cold northern, tropical highland


@pytest.fixture(scope="module")
def project():
    spec = yaml.safe_load((ROOT / "ml_spec.yaml").read_text())
    raw = ROOT / spec["split"]["data_source"]
    train = ROOT / spec["features"]["files"]["train"]
    if not raw.exists() or not train.exists():
        pytest.skip("the raw data and the feature tables are not in this checkout")
    return load_config(spec), pd.read_csv(raw), pd.read_parquet(train)


def test_rows_reproduce_the_training_feature_table(project):
    config, raw, train = project
    raw = raw[raw["city"].isin(CITIES)]
    built = feature_rows(raw, pd.to_datetime(raw["date"]).max().date(), config)
    inputs = [c for c in feature_columns(config) if c != "city"]
    want = train[train["city"].isin(CITIES)].rename(columns={"date": "target_date"})
    want["target_date"] = pd.to_datetime(want["target_date"])
    got = built.merge(want[["city", "target_date"]], on=["city", "target_date"])
    assert len(got) == len(want) > 0
    key = ["city", "target_date"]
    pd.testing.assert_frame_equal(got.set_index(key).sort_index()[inputs + [config["target"]]],
                                  want.set_index(key).sort_index()[inputs + [config["target"]]],
                                  check_dtype=False)
