"""
Deep learning model builder for sensor forecasting.

Single-step (t+1) forecasting: predicts the next value of a sensor
given a sliding window of past observations.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, GRU, Conv1D, Lambda
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(42)


def build_sensor_lstm(input_shape: tuple,
                      units: tuple = (64, 32),
                      dropout: float = 0.4,
                      learning_rate: float = 1e-3) -> Sequential:
    """
    Build a stacked LSTM model for single-step sensor forecasting.
    Parameters
      input_shape : Shape of input sequences (seq_len, n_features).
      units : Number of units in each LSTM layer.
      dropout : Dropout rate applied after each recurrent layer.
      learning_rate : Adam optimiser learning rate.
    Returns
      model : Compiled LSTM model.
    """
    model = Sequential([LSTM(units[0],
                             return_sequences=True,
                             input_shape=input_shape),
                        Dropout(dropout),
                        LSTM(units[1],
                            return_sequences=False),
                        Dropout(dropout),
                        Dense(1)], name='SensorLSTM')

    model.compile(optimizer=Adam(learning_rate, clipnorm=1.0),
                  loss='mse',
                  metrics=['mae'])
                        
    return model


def build_sensor_gru(input_shape: tuple,
                      units: tuple = (64, 32),
                      dropout: float = 0.4,
                      learning_rate: float = 1e-3) -> Sequential:
    """
    Build a stacked GRU model for single-step sensor forecasting.
    Parameters
      input_shape : Shape of input sequences (seq_len, n_features).
      units : Number of units in each GRU layer.
      dropout : Dropout rate applied after each recurrent layer.
      learning_rate : Adam optimiser learning rate.
    Returns
      model : Compiled GRU model.
    """
    model = Sequential([GRU(units[0],
                             return_sequences=True,
                             input_shape=input_shape),
                        Dropout(dropout),
                        GRU(units[1],
                            return_sequences=False),
                        Dropout(dropout),
                        Dense(1)], name='SensorGRU')

    model.compile(optimizer=Adam(learning_rate, clipnorm=1.0),
                  loss='mse',
                  metrics=['mae'])
    return model


def build_sensor_tcn(input_shape, filters=64, kernel_size=3,
                     dilations=(1,2,4,8), dropout=0.2,
                     learning_rate=1e-3):
    """
    Builds a Temporal Convolutional Network (TCN) for sensor time-series regression.

    The model applies stacked causal Conv1D layers with increasing dilation rates
    to capture short- and long-term temporal dependencies. The final time step
    representation is extracted and passed through a dense layer to produce a
    single regression output.

    Compiled with Adam optimizer (gradient clipping enabled) and optimized for MSE loss.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    for d in dilations:
        x = Conv1D(filters, kernel_size,
                   padding='causal',
                   dilation_rate=d,
                   activation='relu')(x)
        x = Dropout(dropout)(x)
    x = Lambda(lambda t: t[:, -1, :])(x)
    output = Dense(1)(x)

    model = Model(inputs,
                  output,
                  name='SensorTCN')

    model.compile(optimizer=Adam(learning_rate, clipnorm=1.0),
                  loss='mse',
                  metrics=['mae'])
    return model
