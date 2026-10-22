"""The champion's recipe: the tree preprocessor from src.features and LightGBM with the tuned settings.

The defaults match the lightgbm entry of the ML project skills' model_core, which built the champion in
notebooks/08_training.ipynb; the tuned settings come from training.champion.params in ml_spec.yaml.
"""

from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from src.features import make_preprocessor

DEFAULTS = {"n_estimators": 300, "learning_rate": 0.05, "verbose": -1, "random_state": 0, "n_jobs": -1}


def build_model(params: dict, config: dict) -> Pipeline:
    """An unfitted pipeline of the preprocessor and the model. params override the defaults."""
    return Pipeline([("prep", make_preprocessor("tree", config)), ("model", LGBMRegressor(**{**DEFAULTS, **params}))])
