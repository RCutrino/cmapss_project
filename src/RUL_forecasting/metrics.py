import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1,
                             np.exp( d / 10) - 1)
    return np.sum(score)

def evaluate(name, y_true, y_pred, results: dict):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    nasa  = nasa_score(y_true, y_pred)

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'NASA Score': nasa}
    print(f"{name:<25} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  NASA={nasa:.1f}")
