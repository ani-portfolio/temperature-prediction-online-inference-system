"""Monitoring pipeline: joins actuals to the logged predictions and writes scores, drift, and gate reports.

From the system design (docs/system_design.md):
* label_feedback: Daily Hopsworks job joining actuals into the feature group scored_predictions
* monitoring: Hopsworks feature monitoring plus Evidently, as a daily job after label feedback
"""
