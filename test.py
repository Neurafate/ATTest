import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the Excel file
def load_excel_data(file_path):
    # Load Excel data into pandas DataFrame
    df = pd.read_excel(file_path)
    
    # Convert 'Date' and 'Time' columns to numerical if necessary
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).astype('int64') / 10**9  # Convert to Unix timestamp (seconds since epoch)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour + \
                     pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute / 60.0  # Convert to hours

    # Encode all string/categorical columns to numeric
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    # Assume the input features are everything except the 'Anomaly' column
    X = df.drop(columns=['Anomaly'])  # Input features
    y = df['Anomaly']  # Target labels
    
    return X, y, label_encoders

# Preprocess the data
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize data
    X_scaled = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # Reshape for LSTM
    return X_scaled, scaler

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: normal or anomaly
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    # Load the Excel data
    file_path = "enhanced_audit_logs.xlsx"  # Replace with your file path
    X, y, label_encoders = load_excel_data(file_path)
    
    # Preprocess the data
    X_scaled, scaler = preprocess_data(X)
    
    # Build the LSTM model
    model = build_lstm_model((X_scaled.shape[1], 1))

    # Save the best model weights during training
    checkpoint = ModelCheckpoint('lstm_model_weights.keras', monitor='loss', save_best_only=True, mode='min')

    # Train the model
    model.fit(X_scaled, y, epochs=100, batch_size=32, callbacks=[checkpoint])

    # Save the model architecture and weights
    model.save('lstm_model.keras')
    
    # Save the scaler and label encoders
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print("Model, scaler, and label encoders saved.")

if __name__ == "__main__":
    main()
