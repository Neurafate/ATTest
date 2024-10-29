# File: train_isolation_forest.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

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

# Step 2: Prepare Features for the Model
def prepare_features(df, user_encoder=None, vectorizer=None):
    # Encode 'User' column
    if user_encoder is None:
        user_encoder = LabelEncoder()
        df['User'] = user_encoder.fit_transform(df['User'])
    else:
        df['User'] = df['User'].apply(lambda x: user_encoder.transform([x])[0] if x in user_encoder.classes_ else -1)
    
    # Text vectorization for 'Activity Description' using TF-IDF
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X_text = vectorizer.fit_transform(df['Activity Description'])
    else:
        X_text = vectorizer.transform(df['Activity Description'])
    
    # Convert 'Activity Description' vector to DataFrame with feature names
    X_text_df = pd.DataFrame(X_text.toarray(), index=df.index, columns=vectorizer.get_feature_names_out())
    
    # Ensure all column names are strings
    X_text_df.columns = X_text_df.columns.astype(str)
    
    # Concatenate all features into a final feature set
    df.reset_index(drop=True, inplace=True)
    X_text_df.reset_index(drop=True, inplace=True)
    X_final = pd.concat([df.drop(columns=['Activity Description', 'Datetime', 'Reason for change', 'Anomaly']), X_text_df], axis=1)
    
    return X_final, user_encoder, vectorizer

# Step 3: Train the Isolation Forest Model
def train_model(file_path):
    # Load the dataset containing 100% normal behavior
    df = pd.read_csv(file_path)
    
    # Preprocess the dataset
    df = preprocess_data(df)
    
    # Prepare features for training
    X_final, user_encoder, vectorizer = prepare_features(df)
    
    # Train Isolation Forest on the normal dataset
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    isolation_forest.fit(X_final)
    
    # Save the trained model and preprocessing tools to disk
    joblib.dump(isolation_forest, 'isolation_forest_model.joblib')
    joblib.dump(user_encoder, 'user_encoder.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    print("Model and preprocessing tools have been saved.")

if __name__ == "__main__":
    train_model('enhanced_audit_logs1.5k.csv')