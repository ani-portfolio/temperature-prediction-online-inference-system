"""Hand written features for this project. The ml-feature-engineering skill creates this file once and never
changes it, so it is yours to edit.

custom_features(df) runs inside src.features.build_features when the feature config has custom: True, after the
built in row and history steps, on the full timeline (training, gap, and test rows together).

Rules, so the features stay honest:
1. Use only information known at each row's prediction time. Never use the target, or columns the data
   dictionary marks as possibly recorded after the outcome.
2. Do not learn from the data here (no means, medians, or encodings over all rows). Learned steps belong in the
   preprocessor, which is fitted on training rows only.
3. Add each new column to the numeric or categorical list in the notebook's feature settings, or it is not used.
"""

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401


def custom_features(df):
    df = df.copy()
    # Both use only the built in lag and rolling columns, which end at least 2 days before the target date.
    # Day over day change on the last complete day: positive means it was warming.
    df["temp_max_change_1d"] = df["temperature_2m_max_lag_2d"] - df["temperature_2m_max_lag_3d"]
    # Last complete day against its 28 day mean: positive means warmer than usual lately.
    df["temp_max_vs_28d_mean"] = df["temperature_2m_max_lag_2d"] - df["temperature_2m_max_roll_28d_mean_shift_2d"]
    return df
