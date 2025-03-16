import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def predict_new_data(new_data, model_path="xgboost_smote_threshold_0.5.pkl", scaler_path="scaler.pkl", feature_names_path="feature_names.pkl", threshold=0.5):
    """
    Load the model and scaler, preprocess the new data, and return predictions and probabilities.

    Parameters:
    - model_path (str): Path to the saved model file.
    - scaler_path (str): Path to the saved scaler file.
    - new_data (array-like): New data to predict.
    - threshold (float): Decision threshold for classification.

    Returns:
    - predictions (ndarray): Binary predictions (0 or 1).
    - probabilities (ndarray): Predicted probabilities for class 1.
    """
    # Load the model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load the scaler
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Preprocess the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Predict probabilities
    probabilities = model.predict_proba(new_data_scaled)[:, 1]
    
    # Apply the threshold
    predictions = (probabilities > threshold).astype(int)
    
    return predictions, probabilities
