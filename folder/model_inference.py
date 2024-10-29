# inference.py
import data_processing as dp
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_model_and_scalers():
    # Load trained model
    model = tf.keras.models.load_model('lstm_model.keras')

    # Load scaler and label encoders
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    return model, scaler, label_encoders

def preprocess_test_data(file_path, scaler, sequence_length=10):
    # Load and preprocess test data
    X, y, _ = dp.load_excel_data(file_path)
    
    # Convert date columns if necessary
    # Assuming 'date_column' is the name of your date column
    X['date'] = pd.to_datetime(X['date'], dayfirst=True, errors='coerce')
    
    X_scaled = scaler.transform(X)  # Scale using the same scaler from training
    X_scaled = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    # Generate sequences
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    generator = TimeseriesGenerator(X_scaled, y, length=sequence_length, batch_size=1)
    X_seq = np.array([x[0] for x in generator])
    y_seq = np.array([y[0] for x, y in generator])
    
    return X_seq, y_seq

def evaluate_model(model, X_seq, y_seq):
    # Make predictions
    y_pred = (model.predict(X_seq) > 0.5).astype("int32")

    # Calculate metrics
    print("Accuracy:", accuracy_score(y_seq, y_pred))
    print("Classification Report:")
    print(classification_report(y_seq, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_seq, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    # File path for testing data
    test_file_path = "anomalous_sequence_test.csv"  # Replace with your test file path

    # Load model and scaler
    model, scaler, _ = load_model_and_scalers()

    # Preprocess test data
    sequence_length = 10  # Use the same sequence length as during training
    X_seq, y_seq = preprocess_test_data(test_file_path, scaler, sequence_length)

    # Evaluate model on test data
    evaluate_model(model, X_seq, y_seq)

if __name__ == "__main__":
    main()
