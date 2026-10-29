"""Which registered models serve: the champion from its lifecycle_status tag, the candidate in canary or A/B with
its traffic share, and in shadow the challengers and the former champions retiring for less than 28 days."""

import json
from datetime import date, timedelta

import pytest

from src.registry import ServingSet, arm, champion, record, serving_models

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


@pytest.mark.parametrize("status", ["canary", "ab"])
def test_a_canary_or_ab_model_is_the_candidate_with_its_share(status):
    mr = FakeRegistry(FakeModel(1, "champion"), FakeModel(2, {"status": "challenger"}),
                      FakeModel(3, {"status": status, "traffic": 0.25, "salt": "v3", "window": {"cutoff": "2026-09-30"}}))
    serving = serving_models(mr, TODAY)
    assert serving.candidate.version == 3
    assert (serving.traffic, serving.salt) == (0.25, "v3")
    assert versions(serving.shadows) == [2]


def test_no_candidate_serves_nothing_to_it():
    serving = serving_models(FakeRegistry(FakeModel(1, "champion")), TODAY)
    assert (serving.candidate, serving.traffic, serving.salt) == (None, 0.0, None)


def test_two_candidates_is_an_error():
    mr = FakeRegistry(FakeModel(1, {"status": "canary", "traffic": 0.1, "salt": "a", "window": {"cutoff": "2026-09-30"}}),
                      FakeModel(2, {"status": "ab", "traffic": 0.5, "salt": "b", "window": {"cutoff": "2026-09-30"}}))
    with pytest.raises(ValueError, match="1, 2"):
        serving_models(mr, TODAY)


@pytest.mark.parametrize("tag", [{"status": "canary", "salt": "a", "window": {"cutoff": "2026-09-30"}},
                                 {"status": "canary", "traffic": 0, "salt": "a", "window": {"cutoff": "2026-09-30"}},
                                 {"status": "canary", "traffic": 1.5, "salt": "a", "window": {"cutoff": "2026-09-30"}},
                                 {"status": "canary", "traffic": "lots", "salt": "a", "window": {"cutoff": "2026-09-30"}}])
def test_a_candidate_needs_a_share_in_0_to_1(tag):
    with pytest.raises(ValueError, match="traffic"):
        serving_models(FakeRegistry(FakeModel(1, tag)), TODAY)


def test_a_candidate_needs_a_salt():
    with pytest.raises(ValueError, match="salt"):
        serving_models(FakeRegistry(FakeModel(1, {"status": "canary", "traffic": 0.1, "window": {"cutoff": "2026-09-30"}})), TODAY)


@pytest.mark.parametrize("window", [None, {}, {"cutoff": ""}])
def test_a_candidate_needs_its_training_window(window):
    tag = {"status": "canary", "traffic": 0.1, "salt": "a"} | ({} if window is None else {"window": window})
    with pytest.raises(ValueError, match="window"):
        serving_models(FakeRegistry(FakeModel(1, tag)), TODAY)


CITY_DAYS = [(city, date(2026, 1, 1) + timedelta(days=d)) for city in ("London", "Tokyo", "Auckland", "Lima")
             for d in range(1000)]


def test_arm_is_the_same_for_the_same_city_day():
    assert {arm("London", date(2026, 11, 3), "v3", 0.5) for _ in range(5)} in ({"candidate"}, {"champion"})


def test_arm_gives_the_candidate_its_share():
    share = sum(arm(c, d, "v3", 0.1) == "candidate" for c, d in CITY_DAYS) / len(CITY_DAYS)
    assert 0.08 < share < 0.12


def test_raising_the_share_only_moves_city_days_to_the_candidate():
    at_10 = {cd for cd in CITY_DAYS if arm(*cd, "v3", 0.10) == "candidate"}
    at_25 = {cd for cd in CITY_DAYS if arm(*cd, "v3", 0.25) == "candidate"}
    assert at_10 < at_25


def test_a_new_salt_reshuffles_the_city_days():
    a = {cd for cd in CITY_DAYS if arm(*cd, "v3", 0.5) == "candidate"}
    b = {cd for cd in CITY_DAYS if arm(*cd, "v4", 0.5) == "candidate"}
    assert a != b


def test_full_traffic_sends_every_city_day_to_the_candidate():
    assert all(arm(c, d, "v3", 1.0) == "candidate" for c, d in CITY_DAYS[:100])


class SavedModel(FakeModel):
    """A model whose files download to a folder, and whose tag can be written."""

    def __init__(self, version, tag, folder):
        super().__init__(version, tag)
        self.folder = folder

    def download(self):
        return str(self.folder)

    def add_tag(self, name, value):
        self._tags[name] = value


def test_load_reference_reads_the_saved_file_or_none(tmp_path):
    import pandas as pd

    from src.registry import REFERENCE_FILE, load_reference

    assert load_reference(SavedModel(1, "champion", tmp_path)) is None
    pd.DataFrame({"city": ["London"], "target_date": [pd.Timestamp("2026-01-01")], "prediction": [8.0]}).to_parquet(
        tmp_path / REFERENCE_FILE)
    assert load_reference(SavedModel(1, "champion", tmp_path))["prediction"].tolist() == [8.0]


def test_set_live_keeps_the_rest_of_the_tag(tmp_path):
    from src.registry import set_live

    model = SavedModel(2, {"status": "challenger", "gate": {"p_one_sided": 0.01}}, tmp_path)
    set_live(model, {"p_one_sided": 0.2, "days": 10})
    assert record(model) == {"status": "challenger", "gate": {"p_one_sided": 0.01},
                             "live": {"p_one_sided": 0.2, "days": 10}}
