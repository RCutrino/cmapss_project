"""
Autoencoder-based anomaly detector.

The autoencoder learns to reconstruct healthy sensor readings.
Reconstruction error on unseen data serves as the anomaly score —
degraded samples produce higher errors than healthy ones.
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def build_autoencoder(input_dim: int,
                      encoding_dim: int = 4,
                      learning_rate: float = 1e-3) -> Model:
    """
    Build a fully-connected autoencoder for anomaly detection.

    Parameters
      input_dim : Number of input features.
      encoding_dim : Size of the bottleneck layer. Smaller = more compression.
      learning_rate : Adam optimiser learning rate.
    Returns
      model : Compiled autoencoder model.
    """
    inputs  = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(inputs)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    outputs = Dense(input_dim, activation='linear')(decoded)

    model = Model(inputs, outputs, name='Autoencoder')
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    
    return model


def autoencoder_predict(X_train: np.ndarray,
                        X_eval: np.ndarray,
                        input_dim: int,
                        percentile: int = 99,
                        epochs: int = 100,
                        batch_size: int = 256) -> tuple:
    """
    Train autoencoder on healthy data and compute anomaly scores.

    Parameters
      X_train : Healthy training data.
      X_eval : Evaluation data (healthy + degraded).
      input_dim : Number of input features.
      percentile : Percentile of training reconstruction errors used as threshold.
      epochs : Maximum training epochs.
      batch_size : Mini-batch size.
    Returns
      model : Trained autoencoder.
      y_pred : Binary predictions: 1 = anomaly, 0 = normal.
      scores : Continuous anomaly scores (reconstruction errors).
      threshold : Detection threshold derived from training errors.
    """
    model = build_autoencoder(input_dim)
    model.fit(X_train, X_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              callbacks=[EarlyStopping(patience=10,
                                       restore_best_weights=True,
                                       verbose=0)],
              verbose=0)

    train_recon = model.predict(X_train, verbose=0)
    eval_recon = model.predict(X_eval,  verbose=0)

    train_errors  = np.mean((X_train - train_recon) ** 2, axis=1)
    scores = np.mean((X_eval  - eval_recon)  ** 2, axis=1)

    threshold = np.percentile(train_errors, percentile)
    y_pred = (scores > threshold).astype(int)

    return model, y_pred, scores, threshold
