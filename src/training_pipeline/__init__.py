"""Training pipeline: reads point in time correct training data from the feature store and writes a trained model to the model registry.

From the system design (docs/system_design.md):
* training: Hopsworks job creating training data from the feature view
* registry: Hopsworks model registry

The interface: run_training (the train job) registers a candidate and starts run_gate (the gate job), which tags it
challenger or rejected with a one sided Diebold-Mariano test against the champion, or persistence before there is
one. training_window, entry_test, and build_model hold the logic and need no network.
"""

from src.training_pipeline.gate import GateReport, entry_test, run_gate
from src.training_pipeline.model import build_model
from src.training_pipeline.pipeline import TrainingReport, run_training
from src.training_pipeline.window import TrainingWindow, training_window

__all__ = ["GateReport", "TrainingReport", "TrainingWindow", "build_model", "entry_test", "run_gate", "run_training",
           "training_window"]
