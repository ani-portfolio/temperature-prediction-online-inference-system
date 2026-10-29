# Deployment

How code reaches the Hopsworks project `online_inference_v2`, how a new project is started, and how to roll back.

## How a change goes live
1. Work on a branch (`ml/v1`, ...) and open a PR into `main`.
2. CI (`.github/workflows/ci.yml`) runs `uv sync --locked`, `ruff check`, and `pytest -m unit` on Python 3.11 and
   3.13, then a dry run of the deploy that prints what the merge would change in Hopsworks.
3. Merge, then run `.github/workflows/deploy.yml` by hand (Actions, deploy, Run workflow, from `main`): the same
   checks, then `python -m deploy.apply` and `python -m deploy.smoke` under the GitHub environment `prod`. A merge
   alone deploys nothing. Deploys run one at a time.

`deploy.apply` makes the project run exactly one commit:
* the clone at `Resources/temperature-prediction-online-inference-system` is checked out at that commit;
* the environments `weather` (jobs, cloned from `pandas-training-pipeline`) and `weather-inference` (the predictor,
  cloned from `pandas-inference-pipeline`), both Python 3.13, get the `runtime` dependency group from `uv.lock`, only
  when `uv.lock` changed;
* `src/` (without tests) is uploaded as `Resources/deploy/src.zip`, which the predictor imports from;
* every job in `src/*/jobs/jobs.yaml` is created or updated, with its schedule;
* the deployment `temperatureforecast` is created (linked to the newest registered model version, with inference
  logging) or updated, then restarted; while no model is registered (a new project) it is skipped, which is not a
  problem;
* `Resources/deploy/state.json` records the commit, the lock hash, and the time.

It never deletes a job (a job in the clone with no settings is reported as `unmanaged`; delete it by hand) and never
decides which model serves: the predictor serves the model tagged champion, so promotion stays by hand.

`deploy.smoke` runs the feature job (`feature_pipeline_hourly`, scheduled daily at 12:15 UTC) and checks one forecast
against the contract in `docs/system_design.md`. Seconds after the restart the gateway can answer 503 (no healthy
upstream) before the predictor is ready, so a 503 is retried once after a minute; any other error fails at once.
While there is no deployment it prints `bootstrap pending` and
passes without running anything. It accepts the persistence fallback while no model is tagged champion
(`fallback_reason: no_champion`); any other fallback right after the feature run is a problem.

## Starting a new project
A new Hopsworks project has no data, no model, and no deployment. Fill it once, in this order:
1. Set `PROJECT` in `src/platform.py`, and the project's API key as in "The API key" below.
2. Merge to `main` and run the deploy workflow. The deploy creates the jobs and skips the deployment; the smoke check says `bootstrap pending`.
   Both pass.
3. Backfill from your machine, before the feature job's first scheduled run at 12:15 UTC:
   `.venv/bin/python -m src.feature_pipeline.jobs.backfill`. It upserts, so a rerun is safe.
4. Run the job `training_pipeline_train` in Hopsworks. It registers a model and starts the gate job, which tags it
   `challenger` (it beat persistence) or `rejected`.
5. If it is `challenger`, tag it champion (step 1 of "Promoting a model"; there is no previous champion to retire).
   If it is `rejected`, leave it: the deployment serves persistence while you find out why.
6. Actions, deploy, Run workflow. This deploy creates the deployment and starts it; the smoke check now asks it for
   a forecast. With a champion it serves the model; without one, persistence (`serving persistence` in the log).

## Adding a job
Add it to the pipeline's `jobs/jobs.yaml` (`script`, and optionally `args`, `schedule` as a Quartz cron in UTC,
`memory_mb`, `cores`). The entry script starts by putting the repo root on `sys.path`, as
`src/feature_pipeline/jobs/hourly.py` does. Jobs that need files outside git (like `backfill.py`, which reads
`data/raw`) have no entry and run locally.

## Adding a package the jobs need
`uv add <package>` for the project, and `uv add --group runtime <package>` if it runs on Hopsworks. Leave out
`hopsworks` and `great-expectations`: the base environment has matched versions.

## Promoting a model
The predictor reads each model's `lifecycle_status` tag only when it starts, so every change below ends with a
restart (`project.get_model_serving().get_deployment("temperatureforecast")`, then `.stop()` and `.start()`).
1. Tag the new champion `{"status": "champion", ...}`, keeping the rest of its record.
2. Tag the previous champion `{"status": "retiring", "since": "<today>", ...}`: it predicts in shadow for 28 days.
3. Restart the deployment. Between steps 1 and 2 there are two champions, which stops a restart, so do both first.

A new challenger from the gate job is shadowed only after a restart; the gate job prints a reminder.

## Reading the monitoring reports
The job `monitoring_pipeline_daily` (13:30 UTC) writes three feature groups and sends nothing; read them yourself.
* `gate_reports`: one row per shadow model per day against the champion on the same live city-days. A challenger
  with `p_one_sided` under 0.05 supports a promotion; a retiring model with it supports a rollback. The newest row is
  also in the model's tag, as `live`. The test waits for 14 shared days.
* `monitoring_reports`: one row per check per day, with `flagged`. `input_psi` and `prediction_psi` compare the last
  28 days with the same calendar days in training; `performance` is the champion's live MAE, flagged when it is not
  significantly better than persistence (the note holds persistence's MAE and the evaluation's monthly range);
  `served_share` is the share of yesterday's answers from a model, flagged under 95%. A flag is a reason to look,
  not to retrain: start `training_pipeline_train` by hand once you know the cause.
* `scored_predictions`: every model prediction with its actual, for your own queries.

Turn on the platform's job failure alerts (email) for every job: a failed monitoring step fails the job, and a job
that stops running leaves old reports that look fine. A model registered before the reference predictions has none,
so its `prediction_psi` row says `no_reference`.

## Rolling back
* A model: tag the bad champion `rejected` (or `retiring` with today's date), tag the previous one `champion`, then
  restart the deployment.
* Code, usually: `git revert` the change on `main`; the revert deploys like any merge.
* Code, quickly: Actions, deploy, Run workflow, with `ref` set to an earlier commit of `main`.

The predictor keeps serving the champion, with its fallback, while a deploy is broken. With no model tagged
champion it serves persistence (`fallback_reason: no_champion`).

## The API key
CI uses the owner's personal Hopsworks API key, stored as the repository secret `HOPSWORKS_API_KEY`; locally it is
read from `.env`. Rotating it means updating both: create a new key in Hopsworks, put it in `.env`, run
`gh secret set HOPSWORKS_API_KEY`, then revoke the old one.

## Simulated traffic
Only requested cities get predictions, so a launchd agent on the owner's laptop runs the simulator every hour while
the laptop is on and logged in: `python -m src.inference_pipeline.jobs.simulate --max-requests 60`, 0 to 60
requests for random cities at random times in each hour, from the working tree and with the key in `.env`. It is
`~/Library/LaunchAgents/com.ani.weather-simulator.plist` (`StartInterval` 3600, `RunAtLoad`, `PYTHONUNBUFFERED=1`,
`WorkingDirectory` the repo root), logging to `~/Library/Logs/weather-simulator.log`, and is not in git.
* Start: `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ani.weather-simulator.plist`
* Stop: `launchctl bootout gui/$(id -u)/com.ani.weather-simulator`
* Try a plan without sending: `.venv/bin/python -m src.inference_pipeline.jobs.simulate --max-requests 60 --dry-run`

## Accepted risks
* There is only prod, and CI runs unit tests only: new code first meets real Hopsworks in a deploy. The smoke check
  and the predictor's fallback limit the damage.
* The CI key has every scope of the owner's account and is shared with local work.
* Local development is on Python 3.11, Hopsworks runs 3.13; CI tests both.

## Inference logging
A spike on 2026-10-23 (a throwaway echo predictor, since deleted) confirmed on Hopsworks Serverless:
* A deployment created with `inference_logger=InferenceLogger()` logs `ALL` to a Kafka topic made for it
  (`<deployment>-inf<NNNN>`). A plain dict such as `{"mode": "ALL"}` is read as no logging.
* Saving a deployment with a new `InferenceLogger()` makes a new topic, and the log job reads only the current one, so
  requests it had not read yet stay behind in the old topic. Hopsworks keeps only the script's file name, so every
  deploy saves the deployment (which also copies an edited `predictor.py` in); `deploy.apply` passes the current
  topic's name each time to keep it. Found on 2026-10-29, after five deploys had left five topics
  (`temperatureforecast-inf3697`, `inf8884`, `inf9490`, `inf4729`, `inf0497`).
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
