import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
np.random.seed(42)
tf.random.set_seed(42)


def evaluate_forecast(model_name: str,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sensor: str,
                      results: dict,
                      model = None,
                      training_time: float = 0.0):
    """
    Evaluate the model and save metrics and computational costs.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    params = model_complexity(model)

    results.setdefault(sensor, {})

    results[sensor][model_name] = {"RMSE": rmse,
                                   "MAE": mae,
                                   "R2": r2,
                                   "train_time": training_time,
                                   "params": params}


class DelayedEarlyStopping(Callback):
    """
    EarlyStopping that activates only after start_epoch epochs,
    preventing premature stopping during the initial learning phases.
    """
    def __init__(self, start_epoch, monitor='val_loss',
                 patience=10, restore_best_weights=True, verbose=0):
        super().__init__()
        self.start_epoch          = start_epoch
        self.monitor              = monitor
        self.patience             = patience
        self.restore_best_weights = restore_best_weights
        self.verbose              = verbose
        self.best_weights         = None
        self.best                 = np.inf
        self.wait                 = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if epoch < self.start_epoch:
            return
        if current < self.best:
            self.best         = current
            self.wait         = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights:
                    self.model.set_weights(self.best_weights)
                if self.verbose:
                    print(f'\nEarlyStopping at epoch {epoch+1}')



def build_callbacks():
  callbacks = [DelayedEarlyStopping(start_epoch=10,
                                    monitor='val_loss',
                                    patience=15,
                                    restore_best_weights=True,
                                    verbose=0),
              ReduceLROnPlateau(monitor='val_loss',
                                factor=0.5,
                                patience=7,
                                min_lr=1e-6,
                                verbose=0)]
  return callbacks


def _minmax(val, all_vals):
    lo, hi = min(all_vals), max(all_vals)
    return 0.0 if hi == lo else (val - lo) / (hi - lo)


def compute_score(results_sensor:dict,
                  model_name: str,
                  w_rmse=0.7,
                  w_time=0.2,
                  w_params=0.1):
    """
    Composite score for NN architecture selection — lower is better.
    Min-max normalization across all models for the same sensor.
    """
    rmse_vals  = [v['RMSE']       for v in results_sensor.values()]
    time_vals  = [v['train_time'] for v in results_sensor.values()]
    param_vals = [v['params']     for v in results_sensor.values()]

    m = results_sensor[model_name]
    return round(w_rmse * _minmax(m['RMSE'], rmse_vals) +
                 w_time * _minmax(m['train_time'], time_vals) +
                 w_params * _minmax(m['params'], param_vals),4)


def model_complexity(model):
    
    if model is None:
        return 0

    # Keras / TensorFlow NN (Model or Sequential)
    if isinstance(model, (Model, Sequential)):
        return model.count_params()

    # PyTorch NN (if any)
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters())

    # XGBoost
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        return len(booster.get_dump())  # number of trees

    return 0 # Default to 0 if not recognized, to prevent NoneType errors


