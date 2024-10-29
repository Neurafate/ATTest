# data_processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def load_excel_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Convert 'Date' column to Unix timestamp (seconds since epoch)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').astype('int64') / 10**9

    # Convert 'Time' column to decimal hours
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').apply(lambda x: x.hour + x.minute / 60.0 + x.second / 3600.0)

    # Encode categorical columns
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    # Select only numeric columns for the model input
    X = df.drop(columns=['Anomaly']).select_dtypes(include=[np.number])
    y = df['Anomaly']

    return X, y, label_encoders

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize data
    X_scaled = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # Reshape for LSTM
    return X_scaled, scaler
