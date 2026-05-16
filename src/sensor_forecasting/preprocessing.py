"""
Data preparation utilities for sensor forecasting.
"""

import pandas as pd


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
