# visualizations.py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import model_inference as inference

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def evaluate_model(file_path):
    predictions, y_test = inference.predict_anomalies(file_path)  # Now prints anomalies
    accuracy = accuracy_score(y_test, (predictions > 0.5).astype(int))
    print(f'Accuracy: {accuracy:.2f}')
    plot_confusion_matrix(y_test, predictions)
