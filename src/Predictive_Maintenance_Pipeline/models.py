import pandas as pd
import numpy as np


def select_best_rul_model(models: dict,
                          X_test: pd.DataFrame,
                          y_true: np.ndarray,
                          seq_data: dict = None) -> tuple:
    """
    Evaluate all RUL models on test data and return the best one.

    For tabular models (RF, XGBoost): predictions are made directly on X_test (last available cycle per engine).

    For Keras models (LSTM): predictions are made on sliding window sequences.
    Since some engines may have fewer cycles than seq_len, they produce no sequences 
    — their prediction is set to NaN and excluded from metric computation. 
    For engines with sequences, only the last sequence prediction is used (closest to end of life).

    Parameters
        models : Dictionary {model_name: fitted_model}.
        X_test : Tabular features — last available cycle per engine.
            Used by RF and XGBoost.
        y_true : Ground truth RUL values — one per engine, same order as test_df['engine_id'].unique().
        seq_data : Required for Keras models. 
          Keys:
            - 'X' : np.ndarray of shape (n_seqs, seq_len, n_features)
            - 'engines' : np.ndarray of engine_id for each sequence
            - 'all_engines' : np.ndarray of all engine_ids in y_true order
    Returns
        best_name : Name of the best model by RMSE.
        best_model : The best fitted model.
        results : RMSE and MAE for each successfully evaluated model.
    """
    results = {}

    for name, model in models.items():
        try:
            is_keras = 'keras' in str(type(model)).lower()

            if is_keras and seq_data is not None:
                X_seq = seq_data['X']
                engines = seq_data['engines']
                all_engines = seq_data['all_engines']
                all_preds  = model.predict(X_seq, verbose=0).flatten()

                preds = np.full(len(all_engines), np.nan)
                for i, e in enumerate(all_engines):
                    idx = np.where(engines == e)[0]
                    if len(idx) > 0:
                        preds[i] = all_preds[idx[-1]]
                preds = np.clip(preds, 0, 125)
            else:
                preds = np.array(model.predict(X_test)).flatten().clip(0, 125)

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
