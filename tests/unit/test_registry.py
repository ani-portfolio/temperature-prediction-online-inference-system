"""Which registered models serve: the champion from its lifecycle_status tag, and in shadow the challengers and the
former champions retiring for less than 28 days."""

import json
from datetime import date

import pytest

from src.registry import ServingSet, champion, record, serving_models

TODAY = date(2026, 11, 2)


class FakeModel:
    def __init__(self, version, tag):
        self.version = version
        self._tags = {} if tag is None else {"lifecycle_status": tag if isinstance(tag, str) else json.dumps(tag)}

    def get_tags(self):
        return self._tags


class FakeRegistry:
    def __init__(self, *models):
        self.models = list(models)

    def get_models(self, name):
        return self.models


def versions(models):
    return [m.version for m in models]


def test_record_reads_json_and_a_bare_word():
    assert record(FakeModel(1, {"status": "challenger", "gate": {}}))["status"] == "challenger"
    assert record(FakeModel(1, "champion")) == {"status": "champion"}
    assert record(FakeModel(1, None)) == {}


def test_two_champions_is_an_error():
    with pytest.raises(ValueError, match="1, 2"):
        champion(FakeRegistry(FakeModel(1, "champion"), FakeModel(2, {"status": "champion"})))


def test_the_champion_serves_and_challengers_shadow():
    mr = FakeRegistry(FakeModel(1, "champion"), FakeModel(2, {"status": "challenger"}),
                      FakeModel(3, {"status": "rejected"}), FakeModel(4, {"status": "candidate"}),
                      FakeModel(5, {"status": "challenger"}))
    serving = serving_models(mr, TODAY)
    assert isinstance(serving, ServingSet)
    assert serving.champion.version == 1
    assert versions(serving.shadows) == [2, 5]


def test_a_retiring_model_shadows_for_28_days():
    mr = FakeRegistry(FakeModel(2, "champion"),
                      FakeModel(1, {"status": "retiring", "since": "2026-10-06"}),    # 27 days ago
                      FakeModel(0, {"status": "retiring", "since": "2026-10-05"}))    # 28 days ago: retired
    assert versions(serving_models(mr, TODAY).shadows) == [1]


def test_a_retiring_model_without_a_date_is_an_error():
    with pytest.raises(ValueError, match="since"):
        serving_models(FakeRegistry(FakeModel(1, {"status": "retiring"})), TODAY)


def test_no_champion_still_has_shadows():
    serving = serving_models(FakeRegistry(FakeModel(1, {"status": "challenger"})), TODAY)
    assert serving.champion is None
    assert versions(serving.shadows) == [1]
