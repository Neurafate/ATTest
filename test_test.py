import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Function to flag unseen labels as anomalies
def flag_unseen_labels_as_anomalies(df, label_encoders):
    anomaly_flags = np.zeros(len(df), dtype=int)  # Initialize anomaly flags as all zeros (no anomalies)

    # For each categorical column, check if there are unseen labels
    for column in df.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            known_labels = set(label_encoders[column].classes_)
            unseen_mask = ~df[column].isin(known_labels)  # Find unseen labels
            anomaly_flags[unseen_mask] = 1  # Flag rows with unseen labels as anomalies
        else:
            print(f"No label encoder found for column: {column}")

    return anomaly_flags

# Load the Excel file and preprocess the data
def load_and_preprocess_data(file_path, scaler, label_encoders=None):
    # Load the data
    df = pd.read_excel(file_path)
    
    # Convert 'Date' and 'Time' columns to numerical if necessary
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).astype('int64') / 10**9  # Convert to Unix timestamp (seconds since epoch)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour + \
                     pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute / 60.0  # Convert to hours

    # Encode categorical columns using provided label encoders and flag unseen labels
    anomaly_flags = flag_unseen_labels_as_anomalies(df, label_encoders)
    
    # Encode the known labels with the label encoders
    if label_encoders:
        for column in df.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                df[column] = df[column].map(lambda s: label_encoders[column].transform([s])[0] if s in label_encoders[column].classes_ else np.nan)
    
    # Drop rows that contain NaNs (unseen labels)
    df_clean = df.dropna(subset=df.select_dtypes(include=[np.number]).columns)

    # Check if any valid data remains after dropping NaNs
    if df_clean.empty:
        print("No valid rows remaining after filtering unseen labels. Returning only anomaly flags.")
        return None, df, anomaly_flags

    # Preprocess using the passed scaler (StandardScaler) and reshape for LSTM
    X_clean = df_clean.drop(columns=['Anomaly'], errors='ignore')  # Exclude 'Anomaly' column if present
    X_scaled = scaler.transform(X_clean)  # Standardize data using the provided scaler
    X_scaled = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # Reshape for LSTM
    
    return X_scaled, df_clean, anomaly_flags

# Main function to load the model, preprocess the new data, and detect anomalies
def detect_anomalies(file_path, model_path, scaler_path, label_encoders_path):
    # Load the trained LSTM model
    model = load_model(model_path)
    
    # Load the saved scaler and label encoders
    scaler = pd.read_pickle(scaler_path)
    label_encoders = pd.read_pickle(label_encoders_path)
    
    # Load and preprocess the new sequence data
    X_scaled, original_df, anomaly_flags = load_and_preprocess_data(file_path, scaler, label_encoders)
    
    # If no valid data remains (i.e., all rows are flagged as anomalies)
    if X_scaled is None:
        original_df['Predicted Anomaly'] = anomaly_flags  # Set all flagged rows as anomalies
        output_file = "anomaly_predictions_with_unseen.xlsx"
        original_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
        return original_df
    
    # Use the model to predict anomalies for rows not flagged due to unseen labels
    predictions = model.predict(X_scaled)
    
    # Create a new column in the original dataframe to indicate predicted anomalies from the model
    original_df['Predicted Anomaly'] = (predictions > 0.5).astype(int)  # Thresholding at 0.5 for binary classification
    
    # Combine model predictions with unseen label flags (if a row is flagged as unseen, set anomaly to 1)
    original_df['Predicted Anomaly'] = np.maximum(original_df['Predicted Anomaly'], anomaly_flags)

    # Output the results
    print("Predictions completed. Lines with anomalies (1) and normal (0):")
    
    # Save the results back to an Excel file or display them
    output_file = "anomaly_predictions_with_unseen.xlsx"
    original_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
    return original_df

# Run the anomaly detection
if __name__ == "__main__":
    # Paths to the necessary files (replace with actual file paths)
    file_path = "anomalous_sequence_5.csv"  # The new data file to check
    model_path = "lstm_model.keras"  # Trained LSTM model
    scaler_path = "scaler.pkl"  # The scaler used during training, saved as a pickle file
    label_encoders_path = "label_encoders.pkl"  # Label encoders used during training, saved as a pickle file
    
    # Call the function to detect anomalies
    results = detect_anomalies(file_path, model_path, scaler_path, label_encoders_path)

    # Display the results (optional)
    print(results[['Predicted Anomaly']])
