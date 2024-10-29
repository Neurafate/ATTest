# main.py
import data_processing as dp
import model_training as mt
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

def main():
    # Load and preprocess data
    file_path = "enhanced_audit_logs.csv"  # Replace with your file path
    X, y, label_encoders = dp.load_excel_data(file_path)
    X_scaled, scaler = dp.preprocess_data(X)
    
    # Experiment with different sequence lengths
    sequence_length = 10  # Start with 10, try different lengths (e.g., 20, 30)
    X_seq, y_seq = mt.create_sequences(X_scaled, y, sequence_length)

    # Build the LSTM model
    model = mt.build_lstm_model((X_seq.shape[1], X_seq.shape[2]))

    # Save the best model weights during training
    checkpoint = ModelCheckpoint('lstm_model_weights.keras', monitor='loss', save_best_only=True, mode='min')

    # Train the model
    model.fit(X_seq, y_seq, epochs=100, batch_size=32, callbacks=[checkpoint])

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
