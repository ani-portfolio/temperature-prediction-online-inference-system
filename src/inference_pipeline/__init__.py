"""Inference pipeline: reads new features and a trained model and writes predictions to their consumer.

From the system design (docs/system_design.md):
* inference: Hopsworks KServe predictor reading the online feature view
* serving: Hopsworks model deployment (KServe) with a predictor script
* failure_handling: Fallback in the predictor script
"""
