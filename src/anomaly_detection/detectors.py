"""
Statistical and classical ML anomaly detectors.

All detectors follow the same interface:
- trained exclusively on healthy data
- evaluated on the full healthy/degraded dataset
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def zscore_predict(X_train: np.ndarray,
                   X_eval: np.ndarray,
                   threshold: float = 3.0) -> np.ndarray:
    """
    Flag anomalies using per-sensor Z-score deviation.

    A sample is flagged as anomalous if any sensor reading deviates
    more than `threshold` standard deviations from the training mean.

    Parameters
      X_train : Healthy training data used to compute mean and std.
      X_eval : Evaluation data (healthy + degraded).
      threshold : Z-score threshold above which a sample is flagged.
    Returns
      y_pred Binary predictions: 1 = anomaly, 0 = normal.
    """
    mean = X_train.mean()
    std  = X_train.std()
    z    = np.abs((X_eval - mean) / std)
    
    return (z.max(axis=1) > threshold).astype(int)


def isolation_forest_predict(X_train: np.ndarray,
                             X_eval: np.ndarray) -> tuple:
    """
    Detect anomalies using Isolation Forest.

    Parameters
      X_train : Healthy training data.
      X_eval : Evaluation data (healthy + degraded).
    Returns
      model : Fitted model.
      y_pred : Binary predictions: 1 = anomaly, 0 = normal.
      scores : Continuous anomaly scores (higher = more anomalous).
    """
    model = IsolationForest(n_estimators=200,
                            contamination=0.01,
                            random_state=42,
                            n_jobs=-1)
    model.fit(X_train)
    y_pred = (model.predict(X_eval) == -1).astype(int)
    scores = -model.score_samples(X_eval)
    
    return model, y_pred, scores
