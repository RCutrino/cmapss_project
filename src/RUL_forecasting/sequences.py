"""
Sequence preparation for deep learning models.
Builds 3D arrays of shape (samples, timesteps, features) from
time-series DataFrames using a sliding window approach.
Windows are built per engine to avoid mixing cycles across units.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def make_sequences(df: pd.DataFrame,
                   sensors: list,
                   target: str,
                   seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences from a time-series DataFrame.
    Parameters
        df : DataFrame containing engine_id, cycle, sensor columns and target.
        sensors : Sensor columns to use as input features.
        target : Target column name (e.g. 'RUL').
        seq_len : Length of each sliding window (number of timesteps).
    Returns
        X : Input sequences.
        y : Target value at the last timestep of each window.
    Notes
        Engines with fewer than seq_len cycles are skipped entirely.
    """
    X, y = [], []

    for engine in df['engine_id'].unique():
        eng = (df[df['engine_id'] == engine].sort_values('cycle'))
        vals = eng[sensors].values
        ruls = eng[target].values

        if len(vals) < seq_len:
            continue

        for i in range(len(vals) - seq_len + 1):
            X.append(vals[i : i + seq_len])
            y.append(ruls[i + seq_len - 1])

    return np.array(X), np.array(y)
