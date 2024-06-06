import streamlit as st
import numpy as np
import pickle
import os

# Load the machine learning model
model_path = 'heart_disease_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Streamlit app
st.title("Heart Disease Prediction")

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
input_data = np.array([[age, gender_map[gender], height, weight, ap_hi, ap_lo,
                        cholesterol_map[cholesterol], glucose_map[glucose], 
                        yes_no_map[smoke], yes_no_map[alco], yes_no_map[active]]])

# Predict button
if 'model' in locals():
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

# Dark mode toggle
if st.checkbox('Dark mode'):
    st.markdown("""
        <style>
        body {
            background-color: #1c1c1c;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
