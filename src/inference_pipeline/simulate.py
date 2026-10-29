"""Simulated traffic for the deployed predictor: random single-city requests at random times, by default at most 10 an hour.

Only requested cities get predictions, so the live gate and monitoring depend on traffic. Every request goes through
the deployment like a real one, so the inference logger and the log job record it, and the champion and every shadow
model are scored on the same city-days. Requests in a city's awaiting_today gap get the persistence fallback, as real
traffic would.
"""

from __future__ import annotations

import random

from src.download_weather import CITIES
from src.inference_pipeline.forecast import contract_problems

CITY_NAMES = tuple(name for name, *_ in CITIES)
MAX_REQUESTS = 10   # per hour
MINUTES = 55        # the hourly run ends before the next one starts


def plan_hour(rng: random.Random, cities=CITY_NAMES, max_requests=MAX_REQUESTS,
              minutes=MINUTES) -> list[tuple[float, str]]:
    """The (seconds from now, city) pairs for one hour of traffic, sorted by time: 0 to max_requests requests,
    each for a city drawn with replacement at a uniform time in the window."""
    n = rng.randint(0, max_requests)
    return sorted((rng.uniform(0, minutes * 60), rng.choice(cities)) for _ in range(n))


def send(deployment, city: str) -> tuple[dict, list[str]]:
    """One single-city request, with the answer and its contract problems (empty when it is fine). A failed request
    is a problem, not an exception, so the rest of the hour still runs."""
    try:
        response = deployment.predict(inputs=[{"city": city}])["predictions"][0]
    except Exception as e:   # noqa: BLE001 - a KServe error reaches the client as any exception
        return {}, [f"{city}: request failed: {e}"]
    return response, contract_problems(response, city)
