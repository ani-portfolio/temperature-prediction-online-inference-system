"""The traffic simulator: python -m src.inference_pipeline.jobs.simulate [--seed N] [--max-requests N] [--dry-run]

Run every hour by a launchd agent on the owner's laptop (see 'Simulated traffic' in docs/deployment.md). Plans the hour, waits for each request's time, asks the deployment
for one city, and prints one line per request. Exits 1 when any answer breaks the contract or a request fails."""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root, as in the other job scripts

from src import platform
from src.inference_pipeline.simulate import MAX_REQUESTS, plan_hour, send


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="plan the same hour again")
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS, help="the most requests in the hour")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without sending anything")
    args = parser.parse_args(argv)
    plan = plan_hour(random.Random(args.seed), max_requests=args.max_requests)
    print(f"{len(plan)} requests planned")
    if args.dry_run:
        print("\n".join(f"+{t / 60:5.1f} min  {city}" for t, city in plan))
        return 0
    deployment = platform.login().get_model_serving().get_deployment(platform.DEPLOYMENT)
    if deployment is None:
        print(f"PROBLEM: no deployment {platform.DEPLOYMENT}")
        return 1
    start, problems = time.monotonic(), []
    for t, city in plan:
        time.sleep(max(0.0, start + t - time.monotonic()))
        response, found = send(deployment, city)
        problems += found
        print(f"+{t / 60:5.1f} min  {city:<16} model {response.get('model_version')}  "
              f"fallback {response.get('fallback_reason') or '-'}" + "".join(f"  PROBLEM: {p}" for p in found))
    print(f"{len(plan)} requests sent, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
