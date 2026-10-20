# 16_weather_prediction

Forecast daily temperature for the coming days from historical weather data.

## Setup

The environment uses Python 3.11.8 and is managed with [uv](https://docs.astral.sh/uv/). To rebuild it on a new machine:

```bash
uv sync
.venv/bin/python -m ipykernel install --user --name 16_weather_prediction --display-name "16_weather_prediction (.venv, Python 3.11.8)"
```

In VS Code, open this folder on its own (File, Open Folder) so it picks up `.venv` from `.vscode/settings.json`. In a notebook, choose the "16_weather_prediction (.venv, Python 3.11.8)" kernel if it is not already selected.

Dependencies are listed in `pyproject.toml`, and `uv.lock` pins every version, so `uv sync` rebuilds the same environment. Add a package with `uv add <package>`.

## Project structure

```
ml_spec.yaml       shared project spec: framing, data findings, and decisions
data/raw/       input data, read only (not committed)
data/processed/ data written by the project (not committed)
notebooks/      numbered stage notebooks
src/            reusable code, importable as `src`
models/         fitted models (not committed)
docs/           business framing summary, data dictionary, model card
```

## Workflow

The project follows the ML project skills: ml-project-setup, ml-business-framing (the mode and two anchors),
ml-data-discovery, ml-business-framing again (the rest, drafted from the data), ml-problem-framing, ml-eda-report,
and then ml-train-test-split, ml-feature-engineering, ml-baseline-model,
ml-model-training, and ml-offline-evaluation. Each step writes one numbered notebook (`01_business_framing.ipynb`
to `09_evaluation.ipynb`) that holds its reasoning, decisions, and checks. Running a notebook records its
decisions in `ml_spec.yaml`. Run `/ml-status` in Claude Code to see progress.

## Data

Describe each data source here: where it came from, when it was pulled, and what one row means.
