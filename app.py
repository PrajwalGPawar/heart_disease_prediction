import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Ensure that page state is initialized
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Function to load model
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    else:
        try:
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

# Main function to run the app
def main():
    # Navigation based on page state
    if st.session_state["page"] == "home":
        run_home_page()
    elif st.session_state["page"] == "prediction":
        run_prediction_page()

def run_home_page():
    st.title("Heart Disease Prediction Project")
    st.image("heart_image.png", use_column_width=True)

    st.write("""
    ## Welcome to the Heart Disease Prediction Project

    This application uses a machine learning model to predict the risk of heart disease based on user-provided health metrics. By entering your information, you can get an indication of your risk level and take appropriate action.

    **Features of this application:**
    - Predict the risk of heart disease based on age, gender, height, weight, blood pressure, cholesterol levels, glucose levels, and lifestyle factors.

    **How to use this application:**
    1. Go to the next page by clicking the "Next Page" button below.
    2. Enter your health metrics and get your heart disease risk prediction.

    **Disclaimer:** This application is for informational purposes only and should not be considered medical advice.
    """)

    if st.button("Next Page"):
        st.session_state["page"] = "prediction"
        st.experimental_rerun()

def run_prediction_page():
    st.title("Heart Disease Prediction")

    st.write("""
    This app predicts the risk of heart disease based on your input parameters.
    Please fill out the information below and click "Predict" to see your risk level.
    """)

    # Input fields
    age = st.number_input("Age (in years)", min_value=0, max_value=120, value=25, step=1)
    gender = st.selectbox("Gender", ("Male", "Female"))
    height = st.number_input("Height (cm)", min_value=0, max_value=250, value=170, step=1)
    weight = st.number_input("Weight (kg)", min_value=0, max_value=250, value=70, step=1)
    ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0, max_value=250, value=120, step=1)
    ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=150, value=80, step=1)
    cholesterol = st.selectbox("Cholesterol", ["Normal (Below 200 mg/dL)", "Above Normal (200-239 mg/dL)", "High (240 mg/dL and above)"])
    glucose = st.selectbox("Glucose", ["Normal (80-160 mg/dL)", "Above Normal (160-199 mg/dL)", "High (200 mg/dL and above)"])
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    alco = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    active = st.selectbox("Physical Activity", ["No", "Yes"])

    # Mapping user inputs to model input
    gender_map = {"Male": 1, "Female": 0}
    cholesterol_map = {"Normal (Below 200 mg/dL)": 1, "Above Normal (200-239 mg/dL)": 2, "High (240 mg/dL and above)": 3}
    glucose_map = {"Normal (80-160 mg/dL)": 1, "Above Normal (160-199 mg/dL)": 2, "High (200 mg/dL and above)": 3}
    yes_no_map = {"No": 0, "Yes": 1}

    # Create input array for the model
    input_data = np.array([[age * 365, height, weight, gender_map[gender], ap_hi, ap_lo,
                           cholesterol_map[cholesterol], glucose_map[glucose], 
                           yes_no_map[smoke], yes_no_map[alco], yes_no_map[active]]])

    # Validate inputs before making a prediction
    if age == 0 or height == 0 or weight == 0 or ap_hi == 0 or ap_lo == 0:
        st.warning("Please enter valid non-zero values for all fields.")
    else:
        # Load model
        model = load_model('heart_disease_model.pkl')
        
        if model:
            if st.button("Predict"):
                try:
                    prediction = model.predict(input_data)
                    if prediction[0] == 1:
                        st.error("Warning: You have a high risk of heart disease.")
                    else:
                        st.success("You have a low risk of heart disease.")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.warning("Model is not loaded. Please ensure the model file is available.")

if __name__ == "__main__":
    main()
