# File: app.py

import streamlit as st
import pandas as pd
import joblib
from test_test import preprocess_data, prepare_features

def predict_anomalies(df):
    # Load preprocessing tools and model
    user_encoder = joblib.load('user_encoder.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    isolation_forest = joblib.load('isolation_forest_model.joblib')
    
    # Preprocess the dataset
    df = preprocess_data(df)
    
    # Prepare features for prediction
    X_final = prepare_features(df, user_encoder, vectorizer)
    
    # Predict anomalies using the loaded model
    predictions = isolation_forest.predict(X_final)
    df['Prediction'] = predictions
    
    # Convert predictions: 1 -> Normal, -1 -> Anomaly
    df['Prediction'] = df['Prediction'].apply(lambda x: 0 if x == 1 else 1)
    
    # Filter rows where Anomaly or Prediction is 1
    result_df = df[(df['Anomaly'] == 1) | (df['Prediction'] == 1)]
    
    return result_df

def main():
    st.title("Anomaly Detection")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the CSV
        df = pd.read_csv(uploaded_file)

        # Predict anomalies
        result_df = predict_anomalies(df)

        # Display the filtered results
        st.subheader("Anomalies Detected")
        st.write(result_df)

if __name__ == "__main__":
    main()
