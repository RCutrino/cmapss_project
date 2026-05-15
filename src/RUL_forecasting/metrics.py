"""
Evaluation metrics for RUL forecasting.
Metrics
- RMSE  : Root Mean Squared Error
- MAE   : Mean Absolute Error
- R²    : Coefficient of determination
- NASA Score : Asymmetric scoring function from the original CMAPSS paper.
              Penalises late predictions (underestimating RUL) more than
              early ones, reflecting the real cost of unexpected failures.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the NASA asymmetric scoring function.
    Parameters
        y_true : Ground truth RUL values.
        y_pred : Predicted RUL values.
    Returns
        Cumulative NASA score. Lower is better.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    score = np.where(d < 0, 
                     np.exp(-d / 13) - 1, 
                     np.exp( d / 10) - 1)
    return round(float(np.sum(score)), 3)


def evaluate(name: str,
             y_true: np.ndarray,
             y_pred: np.ndarray,
             results: dict) -> dict:
    """
    Compute all metrics for a model and store them in results dict.
    Parameters
        name : Model name used as key in results.
        y_true : Ground truth RUL values.
        y_pred : Predicted RUL values.
        results : Dictionary where metrics are accumulated across models.
    Returns
        Updated results dictionary.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    nasa  = nasa_score(y_true, y_pred)

    results[name] = {'RMSE':       rmse,
                     'MAE':        mae,
                     'R²':         r2,
                     'NASA Score': nasa}
    print(f"{name:<25} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  NASA={nasa:.1f}")
    
    return results
