"""Monitoring pipeline: joins actuals to the logged predictions and writes scores, drift, and gate reports.

From the system design (docs/system_design.md):
* label_feedback: daily, joining actuals from weather_daily into the feature group scored_predictions
* monitoring: own PSI (src.drift), live MAE against persistence, and the model-served share, in the same daily job

The interface: run_daily (the daily job) writes scored_predictions, gate_reports (each shadow model against the
champion, also copied to the model's tag as its live block), and monitoring_reports. score, live_gate, drift_rows,
performance_row, and served_share_row hold the logic and need no network. Nothing here retrains or promotes: the owner
reads the reports and acts by hand.
"""

from src.monitoring_pipeline.checks import drift_rows, performance_row, served_share_row
from src.monitoring_pipeline.gate import live_gate
from src.monitoring_pipeline.pipeline import DailyReport, run_daily
from src.monitoring_pipeline.scores import score

__all__ = ["DailyReport", "drift_rows", "live_gate", "performance_row", "run_daily", "score", "served_share_row"]
