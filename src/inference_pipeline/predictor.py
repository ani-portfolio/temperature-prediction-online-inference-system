"""The predictor's I/O around forecast(): load the serving models once, look up each city's feature rows, answer.

Built by jobs/predictor.py, the script the KServe deployment runs. Tags are read at startup, so a promotion, a
rollback, or a new challenger takes effect when the deployment restarts.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, time

from src import platform, registry
from src.inference_pipeline.forecast import Model, forecast, lookup_keys
from src.registry import ServingSet


class Server:
    def __init__(self, project, today: date | None = None, load=registry.load):
        serving = registry.serving_models(project.get_model_registry(), today or datetime.now(UTC).date())
        self.models = ServingSet(serving.champion and Model(serving.champion.version, load(serving.champion)),
                                 [Model(m.version, load(m)) for m in serving.shadows],
                                 serving.candidate and Model(serving.candidate.version, load(serving.candidate)),
                                 serving.traffic, serving.salt)
        self.fv = project.get_feature_store().get_feature_view(*platform.FEATURE_VIEW)
        self.fv.init_serving()

    def _rows(self, keys) -> dict[date, dict]:
        """The feature rows for the keys, by target date; a key with no row comes back with empty values."""
        found = self.fv.get_feature_vectors(
            serving_keys=[{"city": city, "target_date": datetime.combine(day, time())} for city, day in keys],
            return_type="pandas", allow_missing=True)
        return {day: row for (_, day), row in zip(keys, found.to_dict("records"), strict=True)}

    def __call__(self, inputs: list[dict], now: datetime | None = None) -> list[dict]:
        """One response per input {city}. A city that is not served, or has no row to forecast from, gets
        {city, error} without failing the other cities; a feature store that cannot be reached fails the request."""
        now = now or datetime.now(UTC)
        out = []
        for item in inputs:
            city = item.get("city")
            try:
                rows = self._rows(lookup_keys(city, now))
                out.append(forecast(city, now, rows, self.models, str(uuid.uuid4())))
            except LookupError as e:   # KeyError (a city not served) is a LookupError too
                out.append({"city": city, "error": str(e).strip("'\"")})
        return out
