"""The traffic simulator: one hour of random single-city requests, each checked against the serving contract."""

import random

import pytest

from src.inference_pipeline.forecast import contract_problems
from src.inference_pipeline.simulate import CITY_NAMES, plan_hour, send

GOOD = {"city": "Auckland", "target_date": "2026-10-23", "forecast_max_c": 17.4, "model_version": 1,
        "fallback_used": False, "fallback_reason": None, "inputs_as_of": "2026-10-21"}


class Deployment:
    def __init__(self, answer):
        self.answer, self.inputs = answer, []

    def predict(self, inputs):
        self.inputs.append(inputs)
        if isinstance(self.answer, Exception):
            raise self.answer
        return {"predictions": [self.answer]}


@pytest.mark.parametrize("seed", range(20))
def test_an_hour_plans_at_most_ten_requests_sorted_inside_the_window(seed):
    plan = plan_hour(random.Random(seed))
    assert len(plan) <= 10
    assert [t for t, _ in plan] == sorted(t for t, _ in plan)
    assert all(0 <= t < 55 * 60 for t, _ in plan)
    assert {c for _, c in plan} <= set(CITY_NAMES)


def test_the_same_seed_plans_the_same_hour():
    assert plan_hour(random.Random(7)) == plan_hour(random.Random(7))


def test_over_many_hours_the_plan_uses_every_count_up_to_the_cap():
    counts = {len(plan_hour(random.Random(seed))) for seed in range(500)}
    assert counts == set(range(11))


def test_a_fresh_model_answer_has_no_problems_and_asks_for_one_city():
    deployment = Deployment(GOOD)
    response, problems = send(deployment, "Auckland")
    assert (response, problems) == (GOOD, [])
    assert deployment.inputs == [[{"city": "Auckland"}]]


@pytest.mark.parametrize("reason", ["awaiting_today", "stale", "seasonal", "model_error", "no_champion"])
def test_a_fallback_the_design_allows_is_not_a_problem(reason):
    _, problems = send(Deployment({**GOOD, "fallback_used": True, "fallback_reason": reason}), "Auckland")
    assert problems == []


def test_an_unknown_fallback_reason_is_a_problem():
    _, problems = send(Deployment({**GOOD, "fallback_used": True, "fallback_reason": "guess"}), "Auckland")
    assert any("fallback_reason" in p for p in problems)


def test_a_missing_field_is_a_problem():
    _, problems = send(Deployment({k: v for k, v in GOOD.items() if k != "inputs_as_of"}), "Auckland")
    assert any("inputs_as_of" in p for p in problems)


def test_a_city_error_is_a_problem():
    _, problems = send(Deployment({"city": "Auckland", "error": "no row to forecast from"}), "Auckland")
    assert any("no row to forecast from" in p for p in problems)


def test_a_failed_request_is_a_problem_not_a_crash():
    response, problems = send(Deployment(RuntimeError("HTTP 500")), "Auckland")
    assert response == {}
    assert any("HTTP 500" in p for p in problems)


def test_the_contract_check_accepts_a_good_response():
    assert contract_problems(GOOD, "Auckland") == []
