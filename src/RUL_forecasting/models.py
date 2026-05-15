"""
Deep learning model builders for RUL forecasting.
All models output a single continuous value (RUL prediction) and are compiled with MSE loss and Adam optimiser.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm(input_shape: tuple,
              units: tuple = (128, 64),
              dropout: float = 0.2,
              learning_rate: float = 1e-3) -> Sequential:
    """
    Build a stacked LSTM model.
    Parameters:
      input_shape : Shape of input sequences (seq_len, n_features).
      units : Number of units in each LSTM layer.
      dropout : Dropout rate applied after each recurrent layer.
      learning_rate : Adam optimiser learning rate.
    Returns
      model : Compiled LSTM model.
    """
    model = Sequential([LSTM(units[0], return_sequences=True, input_shape=input_shape),
                        Dropout(dropout),
                        LSTM(units[1], return_sequences=False),
                        Dropout(dropout),
                        Dense(32, activation='relu'),
                        Dense(1)], name='LSTM')
    model.compile(optimizer=Adam(learning_rate),
                  loss='mse',
                  metrics=['mae'])
    
    return model


def build_bilstm(input_shape: tuple,
                units: tuple = (128, 64),
                dropout: float = 0.2,
                learning_rate: float = 1e-3) -> Sequential:
    """
    Build a stacked Bidirectional LSTM model.
    Parameters
      input_shape : Shape of input sequences (seq_len, n_features).
      units : Number of units in each BiLSTM layer.
      dropout : Dropout rate applied after each recurrent layer.
      learning_rate : Adam optimiser learning rate.
    Returns
      model : Compiled BiLSTM model.
    """
    model = Sequential([Bidirectional(LSTM(units[0], return_sequences=True), input_shape=input_shape),
                        Dropout(dropout),
                        Bidirectional(LSTM(units[1], return_sequences=False)),
                        Dropout(dropout),
                        Dense(32, activation='relu'),
                        Dense(1)], name='BiLSTM')
    model.compile(optimizer=Adam(learning_rate),
                  loss='mse',
                  metrics=['mae'])
        
    return model
