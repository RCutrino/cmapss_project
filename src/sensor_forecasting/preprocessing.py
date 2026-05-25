"""
Data preparation utilities for sensor forecasting.
"""

import pandas as pd
import numpy as np

def add_forecast_targets(df: pd.DataFrame, sensors: list) -> pd.DataFrame:
    """
    Adds t+1 target columns per sensor, grouped by engine.
    Parameters
      df : DataFrame containing engine_id and sensor columns.
      sensors : Sensor columns for which to create t+1 targets.
    Returns
      DataFrame with additional `sensor_X_target` columns.
      Last cycle of each engine is dropped (no target available).
    """
    df = df.copy()
    for s in sensors:
        df[f'{s}_target'] = df.groupby('engine_id')[s].shift(-1)
    df = df.dropna(subset=[f'{s}_target' for s in sensors])
    
    return df.reset_index(drop=True)


def make_forecast_sequences(df: pd.DataFrame,
                            sensors: list,
                            target: str,
                            seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for single-step sensor forecasting.

    Parameters
        df : DataFrame containing engine_id, cycle, sensor columns and target.
        sensors : Sensor columns to use as input features.
        target  : Target column name (e.g. 'sensor_4_target').
        seq_len : Length of each sliding window (number of timesteps).
    Returns
        X : Input sequences of shape (n_samples, seq_len, n_features).
        y : Target value at timestep t+1 for each window.
    Notes
        Engines with fewer than seq_len + 1 cycles are skipped entirely.
        The last cycle of each engine is excluded — no t+1 available.
    """
    X, y = [], []
    for engine in df['engine_id'].unique():
        eng = df[df['engine_id'] == engine].sort_values('cycle')
        vals = eng[sensors].values
        targets = eng[target].values

        if len(vals) < seq_len + 1:
            continue

        for i in range(len(vals) - seq_len):
            X.append(vals[i : i + seq_len])      # window [t-seq_len+1, t]
            y.append(targets[i + seq_len])         # target at t+1

    return np.array(X), np.array(y)
