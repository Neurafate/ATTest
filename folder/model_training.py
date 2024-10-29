# model_training.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))  # Change the output layer based on your use case (e.g., more neurons for multi-class)
    model.compile(loss='mean_squared_error', optimizer='adam')  # Adjust the loss function based on your problem
    return model


def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])  # Capture sequences of the specified length
        y_seq.append(y[i + sequence_length])    # The label corresponding to the last step in the sequence
    
    return np.array(X_seq), np.array(y_seq)
