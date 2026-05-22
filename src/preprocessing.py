import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_cycle = (df.groupby("engine_id")["cycle"].max().reset_index())
    max_cycle.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    
    return df


def normalize_sensors(df, scaler=None):
    """
    Normalize sensor columns using StandardScaler.
    If scaler is None, fit a new one (training). Otherwise apply existing (inference).
    Returns normalized df and the fitted scaler.
    """
    df = df.copy()
    sensor_cols = [col for col in df.columns if "sensor" in col]
    if scaler is None:
        scaler = StandardScaler()
        df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    else:
        df[sensor_cols] = scaler.transform(df[sensor_cols])
    
    return df, scaler


def add_rolling_stats(df, sensor_cols, window_size=5):
    """
    Add rolling mean and std for each sensor column.
    """
    df = df.copy()
    for sensor in sensor_cols:
        df[f"{sensor}_rolling_mean"] = (df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(window_size).mean()))
        df[f"{sensor}_rolling_std"] = (df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(window_size).std()))
    return df


def add_lag_features(df, sensor_cols, lag_steps):
    """
    Add lag features for each sensor column.
    """
    df = df.copy()
    for sensor in sensor_cols:
        for lag in lag_steps:
            df[f"{sensor}_lag_{lag}"] = (df.groupby("engine_id")[sensor]
                .transform(lambda x: x.shift(lag)))
    return df


def build_final_dataset(df, identifier_cols, target_col, selected_sensors):
    """
    Select final features and drop NaN rows.
    """
    rolling_cols = [col for col in df.columns if "rolling" in col]
    lag_cols = [col for col in df.columns if "lag" in col]
    final_cols = identifier_cols + target_col + selected_sensors + rolling_cols + lag_cols
    
    return df[final_cols].dropna()


def split_datasets(df_final, rul_threshold=125, temporal_ratio=0.8, random_seed=42):
    """
    Generate three task-specific splits:
    - RUL Forecasting: 80/20 split by engine_id
    - Sensor Forecasting: 80/20 temporal split per engine
    - Anomaly Detection: healthy (RUL >= threshold) vs degraded (RUL < threshold)
    """
    # RUL Forecasting
    engine_ids = df_final["engine_id"].unique()
    np.random.seed(random_seed)
    train_engines = np.random.choice(engine_ids, size=int(len(engine_ids) * 0.8), replace=False)
    df_rul_train = df_final[df_final["engine_id"].isin(train_engines)]
    df_rul_val = df_final[~df_final["engine_id"].isin(train_engines)]

    # Sensor Forecasting
    def temporal_split(group):
        cutoff = int(len(group) * temporal_ratio)
        return group.iloc[:cutoff], group.iloc[cutoff:]

    train_parts, val_parts = zip(*[temporal_split(g) for _, g in df_final.groupby("engine_id")])
    df_forecast_train = pd.concat(train_parts)
    df_forecast_val = pd.concat(val_parts)

    # Anomaly Detection
    df_anomaly_train = df_final[df_final["RUL"] >= rul_threshold]
    df_anomaly_val = df_final[df_final["RUL"] < rul_threshold]

    return {"rul": (df_rul_train, df_rul_val),
            "forecast": (df_forecast_train, df_forecast_val),
            "anomaly": (df_anomaly_train, df_anomaly_val)}
