"""
Deep learning model builder for sensor forecasting.

Single-step (t+1) forecasting: predicts the next value of a sensor
given a sliding window of past observations.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_sensor_lstm(input_shape: tuple,
                       units: tuple = (64, 32),
                       dropout: float = 0.3,
                       recurrent_dropout = 0.2,
                       learning_rate: float = 1e-3) -> Sequential:
    """
    Build a stacked LSTM model for single-step sensor forecasting.
    Parameters
      input_shape : Shape of input sequences (seq_len, n_features).
      units : Number of units in each LSTM layer.
      dropout : Dropout rate applied after each recurrent layer.
      recurrent_dropout : Dropout rate applied to the recurrent connections.
      learning_rate : Adam optimiser learning rate.
    Returns
      model : Compiled LSTM model.
    """
    model = Sequential([LSTM(units[0], return_sequences=True, 
                             input_shape=input_shape,
                             recurrent_dropout=recurrent_dropout),
                        Dropout(dropout),
                        LSTM(units[1], 
                             return_sequences=False, 
                             recurrent_dropout=recurrent_dropout),
                        Dropout(dropout),
                        Dense(1)], name='SensorLSTM')

    model.compile(optimizer=Adam(learning_rate),
                  loss='mse',
                  metrics=['mae'])
                        
    return model
