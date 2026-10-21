"""Feature pipeline: reads raw data and writes features and labels to the feature store.

From the system design (docs/system_design.md):
* ingestion: Hopsworks job (Python) writing to the feature group weather_daily
* data_validation: Great Expectations suite attached to weather_daily (strict ingestion policy)
* features: Hopsworks feature group weather_features and a feature view (offline and online store)
"""
