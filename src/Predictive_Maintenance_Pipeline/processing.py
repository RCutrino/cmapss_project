import pandas as pd
import numpy as np

CRITICAL = 'CRITICAL'
WARNING = 'WARNING'
NORMAL = 'NORMAL'
CRITICAL_RUL = 30
WARNING_RUL = 60
SCORE_WARNING = 0.55
SCORE_CRITICAL = 0.75
RUL_CAP = 125
FORECAST_WEIGHT = 0.4
ISOLATION_FOREST_WEIGHT = 0.4
AUTOENCODER_WEIGHT = 0.2


def make_sf_sequences(df: pd.DataFrame,
                      sensors: list,
                      seq_len: int) -> tuple:
    """
    Builds sliding-window sequences for sensor forecasting.
    Returns X, engine_ids and last cycles for alignment.
    """
    X, engine_ids, cycles = [], [], []
    for engine in df['engine_id'].unique():
        eng  = df[df['engine_id'] == engine].sort_values('cycle')
        vals = eng[sensors].values
        cyc  = eng['cycle'].values
        if len(vals) < seq_len:
            continue
        for i in range(len(vals) - seq_len + 1):
            X.append(vals[i : i + seq_len])
            engine_ids.append(engine)
            cycles.append(cyc[i + seq_len - 1])
    return np.array(X), np.array(engine_ids), np.array(cycles)


def forecast_anomaly_score(df: pd.DataFrame,
                           best_models: dict,
                           nn_registry: dict,
                           xgb_models: dict,
                           sf_sensors: list,
                           sf_features: list,
                           residual_std: dict,
                           seq_len: int,
                           train_means: dict) -> pd.DataFrame:
    """
    Computes the forecast anomaly score for each row in the DataFrame.
    For each sensor:
      - predicts t+1 value using the best model
      - computes the normalized residual: |actual_t+1 - pred| / residual_std[s]
    The forecast anomaly score is the mean of the normalized residual on all sensors.
    Returns a DataFrame with engine_id, cycle and forecast_score.
    """
    X_seq, eng_ids, cyc_ids = make_sf_sequences(df, sf_sensors, seq_len)
    aligned = pd.DataFrame({'engine_id': eng_ids, 'cycle': cyc_ids})
    for s in sf_sensors:
        best = best_models[s]
        if best == 'Persistence':
            lookup = (df[['engine_id', 'cycle', s]]
                      .set_index(['engine_id', 'cycle']))
            pred = np.array([float(lookup.loc[(e, c), s])
                            for e, c in zip(eng_ids, cyc_ids)])
        elif best == 'XGBoost':
            xgb_df = pd.DataFrame(index=range(len(eng_ids)))
            for i, (e, c) in enumerate(zip(eng_ids, cyc_ids)):
                row = df[(df['engine_id'] == e) & (df['cycle'] == c)]
                xgb_df.loc[i] = row[sf_features].values[0]
            pred = xgb_models[s].predict(xgb_df.values)
        else:  # TCN / GRU / LSTM
            pred = nn_registry[s].predict(X_seq, verbose=0).flatten()
        actual_lookup = {}
        for engine in df['engine_id'].unique():
            eng = df[df['engine_id'] == engine].sort_values('cycle')
            for idx in range(len(eng) - 1):
                key = (engine, eng['cycle'].iloc[idx])
                actual_lookup[key] = eng[s].iloc[idx + 1]
        actual = np.array([actual_lookup.get((e, c), np.nan)
                          for e, c in zip(eng_ids, cyc_ids)])
        resid = np.abs(actual - pred) / (residual_std[s] + 1e-8)
        aligned[f'{s}_norm_resid'] = resid
    resid_cols = [f'{s}_norm_resid' for s in sf_sensors]
    aligned['forecast_score'] = aligned[resid_cols].mean(axis=1)
    return aligned[['engine_id', 'cycle', 'forecast_score']]


def isolation_forest_score(df: pd.DataFrame,
                           iso_forest,
                           sf_sensors: list) -> pd.DataFrame:
    """
    Compute the Isolation Forest anomaly score for each row in the DataFrame.
    Score = -score_samples (higher values indicate higher anomaly).
    Returns a DataFrame with engine_id, cycle, and if_score
    """
    X = df[sf_sensors].values
    scores = -iso_forest.score_samples(X)
    return pd.DataFrame({'engine_id': df['engine_id'].values,
                         'cycle': df['cycle'].values,
                         'if_score': scores})

def autoencoder_score(df: pd.DataFrame,
                      model_ae,
                      features_ae: list) -> pd.DataFrame:
    """
    Compute the reconstruction error of the autoencoder for each row in the DataFrame.
    Score = MSE between input and ricostruction.
    Returns a DataFrame with engine_id, cycle and ae_score.
    """
    X = df[features_ae].values
    recon = model_ae.predict(X, verbose=0)
    scores = np.mean((X - recon) ** 2, axis=1)
    return pd.DataFrame({'engine_id': df['engine_id'].values,
                         'cycle': df['cycle'].values,
                         'ae_score': scores})

def combine_anomaly_scores(forecast_scores_df: pd.DataFrame,
                           if_scores_df: pd.DataFrame,
                           ae_scores_df: pd.DataFrame,
                           w_forecast: float = FORECAST_WEIGHT,
                           w_if: float = ISOLATION_FOREST_WEIGHT,
                           w_ae: float = AUTOENCODER_WEIGHT) -> pd.DataFrame:
    """
    Combines the three anomaly signals in an unique normalized score.
    Weights:
      w_forecast — sensor forecasting residual (calibrated on residual_std)
      w_if — Isolation Forest score
      w_ae — Autoencoder reconstruction error
    Every signals are normalized to [0, 1] before combining.
    """
    df = forecast_scores_df.merge(if_scores_df,
                                  on=['engine_id', 'cycle'], how='inner')
    df = df.merge(ae_scores_df,
                  on=['engine_id', 'cycle'], how='inner')
    # Normalization of all signals
    for col in ['forecast_score', 'if_score', 'ae_score']:
        lo = df[col].min()
        hi = df[col].max()
        df[f'{col}_norm'] = 0.0 if hi == lo else (df[col] - lo) / (hi - lo)
    # Weighted combined score
    df['anomaly_score'] = (w_forecast * df['forecast_score_norm'] +
                           w_if * df['if_score_norm'] +
                           w_ae * df['ae_score_norm'])
    return df[['engine_id', 'cycle',
               'forecast_score', 'if_score', 'ae_score',
               'anomaly_score']]


def make_rul_sequences(df: pd.DataFrame,
                       sensors: list,
                       seq_len: int) -> tuple:
    """
    Builds sliding-window sequences for RUL forecasting.
    Returns X, engine_ids and last cycles for alignment.
    """
    X, engine_ids, last_cycles = [], [], []
    for engine in df['engine_id'].unique():
        eng  = df[df['engine_id'] == engine].sort_values('cycle')
        vals = eng[sensors].values
        cyc  = eng['cycle'].values
        if len(vals) < seq_len:
            continue
        for i in range(len(vals) - seq_len + 1):
            X.append(vals[i : i + seq_len])
            engine_ids.append(engine)
            last_cycles.append(cyc[i + seq_len - 1])
    return np.array(X), np.array(engine_ids), np.array(last_cycles)

def predict_rul(df: pd.DataFrame,
                lstm_rul,
                seq_sensors: list,
                rul_features: list,
                seq_len: int,
                rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Predict the RUL for each engine at the ultimate cycle available.
    Strategy:
      - LSTM: uses the last seq_len sequences available for each engine
      - if the engine has less than seq_len sequences, uses the mean of the
      sequences available as fallback
    Returns a DataFrame with engine_id and RUL_pred.
    """
    X_seq, eng_ids, last_cycs = make_rul_sequences(
        df, seq_sensors, seq_len)
    # LSTM predictions on all the sequences
    all_preds = lstm_rul.predict(X_seq, verbose=0).flatten()
    all_preds = np.clip(all_preds, 0, rul_cap)
    # For each engine: uses the last sequence available
    results = []
    for engine in df['engine_id'].unique():
        mask = eng_ids == engine
        if mask.sum() == 0:
            # Fallback: if none sequence is available uses the mean value
            eng_data = df[df['engine_id'] == engine].sort_values('cycle')
            X_flat = eng_data[rul_features].iloc[[-1]].values
            # the mean of prediction on single row is not possible
            # → uses RUL_cap/2 as conservative fallback
            pred = rul_cap / 2
        else:
            # Last sequence available = most recent cycle
            last_idx = np.where(mask)[0][-1]
            pred = float(all_preds[last_idx])
        results.append({'engine_id': engine, 'RUL_pred': pred})
    return pd.DataFrame(results)


def assign_alert(row: pd.Series,
                 rul_critical: int = CRITICAL_RUL,
                 rul_warning: int = WARNING_RUL,
                 score_critical: float = SCORE_CRITICAL,
                 score_warning: float  = SCORE_WARNING) -> str:
    """
    Assign alert based on RUL and anomaly score.
    Logic:
      CRITICAL → RUL_pred ≤ rul_critical  OR  anomaly_score ≥ score_critical
      WARNING  → RUL_pred ≤ rul_warning   OR  anomaly_score ≥ score_warning
      NORMAL   → altrimenti
    The thresholds are separated and indipendent
    - a strong signal in either channels is enough to escalte the alert.
    """
    if (row['RUL_pred'] <= rul_critical or
            row['anomaly_score'] >= score_critical):
        return CRITICAL
    if (row['RUL_pred'] <= rul_warning or
            row['anomaly_score'] >= score_warning):
        return WARNING
    return NORMAL

def true_alert(rul_true: float,
               rul_critical: int = CRITICAL_RUL,
               rul_warning: int  = WARNING_RUL) -> str:
    """
    Assign alert ground truth based on RUL.
    """
    if rul_true <= rul_critical:
        return CRITICAL
    if rul_true <= rul_warning:
        return WARNING
    return NORMAL


def dominant_signal(row):
    """
    Identifies which signal dominates the alert.
    NORMAL non ha segnale dominante.
    """
    if row['alert'] == 'NORMAL':
        return 'None'
    rul_triggered = row['RUL_pred'] <= (CRITICAL_RUL if row['alert'] == 'CRITICAL' 
                                                     else WARNING_RUL)
    score_triggered = row['anomaly_score'] >= (SCORE_CRITICAL if row['alert'] == 'CRITICAL' 
                                                              else SCORE_WARNING)
    if rul_triggered and score_triggered:
        return 'Both'
    if rul_triggered:
        return 'RUL only'
    return 'Anomaly score only'
