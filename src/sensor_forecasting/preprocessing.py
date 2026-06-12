"""
Data preparation utilities for sensor forecasting.
"""

import pandas as pd
import numpy as np

np.random.seed(42)


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


def split_train_for_es(train_df: pd.DataFrame,
                       es_ratio: float = 0.2) -> tuple:
    """
    Split train_df into:
    - train_fit : first (1-es_ratio)% of cycles per engine → training LSTM
    - train_es  : last es_ratio% of cycles per engine   → EarlyStopping LSTM

    train_df remains unchanged for XGBoost (which uses TimeSeriesSplit internally).
    val_df is now used exclusively for final evaluation.
    """
    fit_parts, es_parts = [], []
    for _, eng in train_df.groupby('engine_id'):
        eng = eng.sort_values('cycle')
        cutoff = int(len(eng) * (1 - es_ratio))
        fit_parts.append(eng.iloc[:cutoff])
        es_parts.append(eng.iloc[cutoff:])
    train_fit = pd.concat(fit_parts).reset_index(drop=True)
    train_es  = pd.concat(es_parts).reset_index(drop=True)
    return train_fit, train_es


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

        if len(vals) < seq_len:
            continue

        for i in range(len(vals) - seq_len + 1):
            X.append(vals[i : i + seq_len])      # window [t-seq_len+1, t]
            y.append(targets[i + seq_len - 1])         # target at t+1

    return np.array(X), np.array(y)


def get_aligned_val(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """
    Returns the rows of df that can be evaluated by all models.
    For each engine, skips the first seq_len-1 cycles — those for which
    the LSTM does not yet have a complete window.
    """
    parts = []
    for _, eng in df.groupby('engine_id'):
        eng = eng.sort_values('cycle')
        if len(eng) < seq_len:
            continue
        parts.append(eng.iloc[seq_len - 1:])
    return pd.concat(parts).reset_index(drop=True)

