# 16_weather_prediction

Forecast daily temperature for the coming days from historical weather data.

## Environment
* Python 3.11.8 lives in `.venv/`. Run code with `.venv/bin/python` (or `uv run python`), never a global `python3`.
* Packages are managed with uv. Add one with `uv add <package>` (or `uv add --dev <package>` for tools such as pytest), which updates `pyproject.toml` and `uv.lock` and installs it. Never `pip install` into `.venv` directly, because `uv sync` would not know about it.
* After pulling changes, run `uv sync --inexact` to match `.venv` to `uv.lock`.
* Never delete or recreate `.venv/`. If it looks broken, stop and tell the user.
* The Jupyter kernel for this project is `16_weather_prediction`, shown in editors as "16_weather_prediction (.venv, Python 3.11.8)". Notebooks in `notebooks/` use this kernel.
* `from src.<module> import <name>` works from notebooks, scripts, and tests in any folder, because the project is installed in `.venv`.

## Layout
* `data/raw/`: input data. Treat it as read only. Not committed to git.
* `data/processed/`: data written by the project, such as split sets and feature tables. Not committed to git.
* `notebooks/`: numbered stage notebooks, plus `00_scratch.ipynb` for free exploration.
* `src/`: reusable code. Move logic out of notebooks into here once it is used twice.
* `models/`: fitted pipelines and models. Not committed to git.
* `tests/`: pytest tests in `unit/` (one function, no network), `component/` (one part of the system, such as the
  feature store or the API, with its real dependencies) and `system/` (end to end, raw data to a served prediction).
  Each folder marks its tests automatically, so `.venv/bin/python -m pytest -m unit` runs one level.
* `docs/`: the business framing summary (`business_framing.md`), the data dictionary (`data_dictionary.yaml`), and later the model card.
* `ml_spec.yaml`: the shared project spec. Committed to git.
* `pyproject.toml`: direct dependencies of the project. `uv.lock` pins every version. Commit both.

## ML workflow
This project uses the ML project skills. `ml_spec.yaml` at the project root is the shared record of every
decision: read it before starting any modelling work, and never contradict it without asking the user.
* Steps, in order: ml-project-setup, ml-business-framing (the mode and two anchors, before the data),
  ml-data-discovery, ml-business-framing again (the rest, drafted from the data), ml-problem-framing, ml-eda-report,
  ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, ml-offline-evaluation,
  ml-system-design (after a go decision: the design of the end to end system).
* Each step writes one numbered notebook in `notebooks/`: `01_business_framing.ipynb`, `02_data_discovery.ipynb`,
  `03_problem_framing.ipynb`, `04_eda_<dataset>.ipynb`, `05_split.ipynb`, `06_features.ipynb`, `07_baseline.ipynb`,
  `08_training.ipynb`, `09_evaluation.ipynb`. The notebook is the record of that step. ml-system-design is the
  exception: it writes `docs/system_design.md` from `answers/11_system.yaml`, with no notebook.
* Cells marked *Decision* hold the answers. To change a decision, edit that cell and run the whole notebook,
  which writes the answers to `ml_spec.yaml`. Do not edit the spec sections those notebooks own by hand.
* The data dictionary is `docs/data_dictionary.yaml`. Check a column's meaning and timing there before using it.
* `answers/` keeps the settings each stage notebook was last built from, so a stage can be rebuilt without
  asking again. Commit it with the notebooks.
* No modelling (split onwards) before the ML framing is confirmed and the targeted EDA has run.
* See progress and the next skill to run with `/ml-status`.
* Each skill writes only its own section of the spec. Open questions in the spec name the stage that needs them.
* Runs of the modelling stages (split onwards, and reviews) are logged to MLflow, in this project's experiment on
  the shared server (`tracking` in `ml_spec.yaml` has the address and the link). The spec decides; MLflow records.
  Start the server with `python3 ~/Projects/claude-skills/tools/mlflow_server.py start` when it is not running.

## Conventions
* Build paths from the project root, not from the current folder. The folders above are also listed under `paths` in `ml_spec.yaml`.
* Keep notebooks short. Functions belong in `src/`.
