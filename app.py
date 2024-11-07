# # File: app.py

# import streamlit as st 
# import pandas as pd
# import joblib
# from test_test import preprocess_data, prepare_features

# def predict_anomalies(df):
#     # Load preprocessing tools and model
#     user_encoder = joblib.load('user_encoder.joblib')
#     vectorizer = joblib.load('vectorizer.joblib')
#     isolation_forest = joblib.load('isolation_forest_model.joblib')
    
#     # Preprocess the dataset
#     df = preprocess_data(df)
    
#     # Prepare features for prediction
#     X_final = prepare_features(df, user_encoder, vectorizer)
    
#     # Predict anomalies using the loaded model
#     predictions = isolation_forest.predict(X_final)
#     df['Prediction'] = predictions
    
#     # Convert predictions: 1 -> Normal, -1 -> Anomaly
#     df['Prediction'] = df['Prediction'].apply(lambda x: 0 if x == 1 else 1)
    
#     # Filter rows where Anomaly or Prediction is 1
#     result_df = df[(df['Anomaly'] == 1) | (df['Prediction'] == 1)]
    
#     return result_df

# def main():
#     st.title("Anomaly Detection")

#     # File uploader
#     uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

#     if uploaded_file is not None:
#         # Load the CSV
#         df = pd.read_csv(uploaded_file)

#         # Predict anomalies
#         result_df = predict_anomalies(df)

#         # Display the filtered results
#         st.subheader("Anomalies Detected")
#         st.write(result_df)

# if __name__ == "__main__":
#     main()


import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Parameters
sequence_length = 10  # Define the sequence length (should match the training sequence length)

# Load the pre-trained model
model = load_model('anomaly_detection_model.h5')

# Function to preprocess the uploaded data
def preprocess_data(data):
    # Drop the 'Reason for change' column
    data = data.drop(columns=['Reason for change'])
    
    # One-Hot Encode the 'User' and 'Activity Description' columns
    data_encoded = pd.get_dummies(data, columns=['User', 'Activity Description'], drop_first=True)
    
    # Separate features and labels
    feature_columns = data_encoded.columns.difference(['Anomaly', 'Date', 'Time'])  # Exclude 'anomaly', 'Date', and 'Time'
    
    # Normalize the features (exclude 'Date' and 'Time')
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_encoded[feature_columns])
    
    # Create sequences for the LSTM model
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i: i + sequence_length])
        return np.array(sequences)
    
    X = create_sequences(data_scaled, sequence_length=sequence_length)  # Use the same sequence length as during training
    return data_encoded, X

# Streamlit app UI
st.title('Anomaly Detection Using LSTM')
st.write('Upload a CSV file for anomaly detection.')

# File upload functionality
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    data = pd.read_csv(uploaded_file)
    
    # Preprocess the data
    data_encoded, X_processed = preprocess_data(data)
    
    # Make predictions using the pre-trained model
    y_pred = (model.predict(X_processed) > 0.5).astype(int)  # Predict and threshold at 0.5
    
    # Add predictions to the original data (or encoded data)
    data_encoded['Prediction'] = np.nan  # Initialize the 'Prediction' column
    data_encoded.loc[data_encoded.index[sequence_length-1:], 'Prediction'] = y_pred  # Assign predictions
    
    # Display the table with the original data and predictions
    anomalies = data_encoded[data_encoded['Anomaly'] == 1]
    st.write("Data with Predictions:")
    st.dataframe(anomalies)  # Displaying the data with predictions
    
    # Optionally, plot the prediction results
    st.write("Prediction Visualization:")
    plt.figure(figsize=(10, 6))
    plt.plot(data_encoded.index[sequence_length-1:], y_pred, label='Predictions')
    plt.title('Anomaly Detection Predictions')
    plt.xlabel('Index')
    plt.ylabel('Anomaly (0 = Normal, 1 = Anomaly)')
    plt.legend()
    st.pyplot()
