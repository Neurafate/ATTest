# File: use_isolation_forest.py

import pandas as pd
import joblib

# Step 1: Preprocess the Data
def preprocess_data(df):
    # Combine 'Date' and 'Time' into a single 'Datetime' column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    # Standardize user identifiers
    df['User'] = df['User'].str.lower().str.strip()
    
    # Clean 'Activity Description' text
    df['Activity Description'] = df['Activity Description'].str.lower().str.replace(r'[^a-z\s]', '', regex=True)
    
    # Extract time-based features
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

# Step 2: Prepare Features for Prediction
def prepare_features(df, user_encoder, vectorizer):
    # Encode 'User' column
    df['User'] = df['User'].apply(lambda x: user_encoder.transform([x])[0] if x in user_encoder.classes_ else -1)
    
    # Text vectorization for 'Activity Description' using TF-IDF
    X_text = vectorizer.transform(df['Activity Description'])
    
    # Convert 'Activity Description' vector to DataFrame with feature names
    X_text_df = pd.DataFrame(X_text.toarray(), index=df.index, columns=vectorizer.get_feature_names_out())
    
    # Ensure all column names are strings
    X_text_df.columns = X_text_df.columns.astype(str)
    
    # Concatenate all features into a final feature set
    df.reset_index(drop=True, inplace=True)
    X_text_df.reset_index(drop=True, inplace=True)
    X_final = pd.concat([df.drop(columns=['Activity Description', 'Datetime', 'Reason for change', 'Anomaly'], errors='ignore'), X_text_df], axis=1)
    
    return X_final

# Step 3: Load the Model and Predict
def predict_anomalies(file_path):
    # Load the test dataset
    df = pd.read_csv(file_path)
    
    # Preprocess the dataset
    df = preprocess_data(df)
    
    # Load preprocessing tools and model
    user_encoder = joblib.load('user_encoder.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    isolation_forest = joblib.load('isolation_forest_model.joblib')
    
    # Prepare features for prediction
    X_final = prepare_features(df, user_encoder, vectorizer)
    
    # Predict anomalies using the loaded model
    predictions = isolation_forest.predict(X_final)
    df['Prediction'] = predictions
    
    # Convert predictions: 1 -> Normal, -1 -> Anomaly
    df['Prediction'] = df['Prediction'].apply(lambda x: 0 if x == 1 else 1)
    
    # Display or save the resulting dataframe with predictions
    df.to_csv('predictions_output.csv', index=False)
    print("Predictions saved to predictions_output.csv.")

if __name__ == "__main__":
    predict_anomalies('combined_output.csv')