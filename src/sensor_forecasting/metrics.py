"""
Evaluation metrics for sensor forecasting.
 
Metrics
-------
- RMSE : Root Mean Squared Error
- MAE  : Mean Absolute Error
- R²   : Coefficient of determination
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping


def evaluate_forecast(model_name, y_true, y_pred, sensor, results):
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3)
    mae  = round(float(mean_absolute_error(y_true, y_pred)), 3)
    r2   = round(float(r2_score(y_true, y_pred)), 3)
    results[sensor][model_name] = {'RMSE': rmse, 
                                   'MAE': mae, 
                                   'R²': r2}
    print(f"{sensor:<12} {model_name:<20} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.3f}")

    return results


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=20, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            super().on_epoch_end(epoch, logs)
