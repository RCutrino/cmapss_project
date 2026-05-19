import pandas as pd
import numpy as np

def select_best_rul_model(models: dict,
                          X_test: pd.DataFrame,
                          y_true: np.ndarray,
                          seq_data: dict = None,
                          seq_len: int = 30) -> tuple:
    """
    Evaluate all RUL models on test data and return the best one.

    Parameters
      models : Dictionary {model_name: fitted_model}.
      X_test : Tabular features for classical ML models.
      y_true : Ground truth RUL values.
      seq_data : {'X': sequences, 'engines': engine_ids} for LSTM evaluation.
      seq_len : Sequence length for LSTM models.
    Returns
      best_name : Name of the best model.
      best_model : The best fitted model.
      results : RMSE and MAE for all models.
    """
    results = {}

    for name, model in models.items():
        try:
            if hasattr(model, 'predict'):
                preds = np.array(model.predict(X_test)).clip(0, 125)
            else:
                continue

            valid = ~np.isnan(preds) & ~np.isnan(y_true)
            rmse  = float(np.sqrt(np.mean((y_true[valid] - preds[valid]) ** 2)))
            mae   = float(np.mean(np.abs(y_true[valid] - preds[valid])))
            results[name] = {'RMSE': round(rmse, 2), 'MAE': round(mae, 2)}
            print(f"{name:<12} RMSE={rmse:.2f}  MAE={mae:.2f}")

        except Exception as e:
            print(f"{name:<12} failed: {e}")

    best_name  = min(results, key=lambda x: results[x]['RMSE'])
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (RMSE={results[best_name]['RMSE']})")

    return best_name, best_model, results
