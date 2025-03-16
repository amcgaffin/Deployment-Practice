import pandas as pd
import streamlit as st
from Inference import predict_new_data  # Import the utility function

# Streamlit App
st.title("Depression Prediction")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
threshold = st.slider("Select Probability Threshold", 0.0, 1.0, 0.5)

if uploaded_file is not None:
    # Read the uploaded file
    new_data = pd.read_csv(uploaded_file)
    
    # Make predictions
    predictions, proba = predict_new_data(
        new_data=new_data,
        threshold=threshold,
        model_path="xgboost_smote_threshold_0.5.pkl",
        scaler_path="scaler.pkl",
        feature_names_path="feature_names.pkl"  # Assuming feature_names.pkl exists and contains the names of the features in the uploaded CSV file
    )
    
    # Add predictions and probabilities to the dataframe
    new_data["Prediction"] = predictions
    new_data["Probability"] = proba

    # Display the results
    st.write("Predictions:")
    st.write(new_data)
    st.download_button("Download Results", new_data.to_csv(index=False), "results.csv")
