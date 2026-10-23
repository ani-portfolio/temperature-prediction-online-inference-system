# Deployment

How code reaches the Hopsworks project `weather_online_inference`, and how to roll it back.

## How a change goes live
1. Work on a branch (`ml/v1`, ...) and open a PR into `main`.
2. CI (`.github/workflows/ci.yml`) runs `uv sync --locked`, `ruff check`, and `pytest -m unit` on Python 3.11 and
   3.13, then a dry run of the deploy that prints what the merge would change in Hopsworks.
3. Merging runs `.github/workflows/deploy.yml`: the same checks, then `python -m deploy.apply` and
   `python -m deploy.smoke` under the GitHub environment `prod`. Deploys run one at a time.

`deploy.apply` makes the project run exactly one commit:
* the clone at `Resources/temperature-prediction-online-inference-system` is checked out at that commit;
* the environment `weather` (cloned from `pandas-training-pipeline`, Python 3.13) gets the `runtime` dependency
  group from `uv.lock`, only when `uv.lock` changed;
* every job in `src/*/jobs/jobs.yaml` is created or updated, with its schedule;
* the predictor restarts, once the inference pipeline has a deployment;
* `Resources/deploy/state.json` records the commit, the lock hash, and the time.

It never deletes a job (a job in the clone with no settings is reported as `unmanaged`; delete it by hand) and
never changes the model version a deployment serves: promotion stays by hand.

`deploy.smoke` runs the hourly feature job and, once there is a deployment, checks one forecast against the
contract in `docs/system_design.md`.

## Adding a job
Add it to the pipeline's `jobs/jobs.yaml` (`script`, and optionally `args`, `schedule` as a Quartz cron in UTC,
`memory_mb`, `cores`). The entry script starts by putting the repo root on `sys.path`, as
`src/feature_pipeline/jobs/hourly.py` does. Jobs that need files outside git (like `backfill.py`, which reads
`data/raw`) have no entry and run locally.

## Adding a package the jobs need
`uv add <package>` for the project, and `uv add --group runtime <package>` if it runs on Hopsworks. Leave out
`hopsworks` and `great-expectations`: the base environment has matched versions.

## Rolling back
* Usually: `git revert` the change on `main`; the revert deploys like any merge.
* Quickly: Actions, deploy, Run workflow, with `ref` set to an earlier commit of `main`.

The predictor keeps serving the champion, with its fallback, while a deploy is broken.

## The API key
CI uses the owner's personal Hopsworks API key, stored as the repository secret `HOPSWORKS_API_KEY`; locally it is
read from `.env`. Rotating it means updating both: create a new key in Hopsworks, put it in `.env`, run
`gh secret set HOPSWORKS_API_KEY`, then revoke the old one.

## Accepted risks
* There is only prod, and CI runs unit tests only: new code first meets real Hopsworks in a deploy. The smoke check
  and the predictor's fallback limit the damage.
* The CI key has every scope of the owner's account and is shared with local work.
* Local development is on Python 3.11, Hopsworks runs 3.13; CI tests both.

## Inference logging
A spike on 2026-10-23 (a throwaway echo predictor, since deleted) confirmed on Hopsworks Serverless:
* A deployment created with `inference_logger=InferenceLogger()` logs `ALL` to a Kafka topic made for it
  (`<deployment>-inf<NNNN>`). A plain dict such as `{"mode": "ALL"}` is read as no logging.
* Each request writes two messages, the request and the response, with the same inference id. Each value is an Avro
  record with no schema registered on the topic, in this field order: modelId (int), modelName (string),
  modelVersion (int), requestTimestamp (long, Unix seconds), responseHttpCode (int), inferenceId (string),
  messageType (string, `org.kubeflow.serving.inference.request` or `.response`), payload (string, the JSON body).
  A nested `log` block in the response arrives intact. Samples: `src/inference_pipeline/tests/unit/fixtures/`.
* A Hopsworks Python job reads the topic with `confluent_kafka` and `project.get_kafka_api().get_default_config()`.
* A predictor script's `Predict.__init__` may only take the parameters `project`, `deployment`, `model`, and
  `async_logger`; anything else, `*args` included, stops the predictor from starting.
* Predictor pods do not run in place: the script is copied to `/mnt/artifacts/`, and HopsFS is **not** mounted
  (`/hopsfs` is empty), so the clone is out of reach. Deploy uploads `src/` as `Resources/deploy/src.zip`, and the
  predictor downloads it at startup and imports from it.
* A deployment needs an environment built on an inference image: `weather` (a training image) is refused. The
  predictor runs in `weather-inference`, cloned from `pandas-inference-pipeline` (Python 3.13, no lightgbm), with the
  same runtime group as `weather`, so models trained in jobs unpickle in the predictor.
