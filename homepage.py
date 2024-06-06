import streamlit as st

# Ensure session state is initialized
if "page" not in st.session_state:
    st.session_state["page"] = "home"

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
