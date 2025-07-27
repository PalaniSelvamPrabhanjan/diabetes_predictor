import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon= "🩺",
    layout="centered"
)

# -----------------------------
# Custom styling (background + status boxes)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)), 
    url(data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=);
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.metric-container {
    background-color: rgba(255,255,255,0.95);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid;
}
.healthy { border-left-color: #22c55e; }
.warning { border-left-color: #f59e0b; }
.danger { border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Page title
# -----------------------------
st.title("Diabetes Risk Predictor")
st.markdown("""
Estimate your likelihood of having diabetes based on key health indicators.  
This tool is for information only and should not replace professional medical advice.
""")
st.markdown("---")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load('HGBCmodel.pkl')
    except FileNotFoundError:
        st.error("Model file 'HGBCmodel.pkl' not found. Make sure it is in the same folder as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# -----------------------------
# Encode user inputs
# -----------------------------
def encode_features(gender, hypertension, heart_disease, smoking_history):
    gender_val = 1 if gender == "male" else 0
    hypertension_val = 1 if hypertension == "positive" else 0
    heart_disease_val = 1 if heart_disease == "positive" else 0
    smoking_map = {
        "No Info": 0,
        "Current": 1,
        "Never": 2,
        "Past": 3
    }
    smoking_val = smoking_map[smoking_history]
    return gender_val, hypertension_val, heart_disease_val, smoking_val

# -----------------------------
# Input form
# -----------------------------
with st.form("diabetes_prediction_form"):
    st.subheader("Your Health Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["male", "female"], index=0)
        hypertension = st.selectbox("Hypertension", ["negative", "positive"], index=0)
        heart_disease = st.selectbox("Heart Disease", ["negative", "positive"], index=0)
        smoking_history = st.selectbox("Smoking History", ["No Info", "Current", "Never", "Past"], index=0)

    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1, format="%.1f")
        if bmi < 18.5:
            st.markdown('<div class="metric-container warning">Underweight</div>', unsafe_allow_html=True)
        elif 18.5 <= bmi <= 24.9:
            st.markdown('<div class="metric-container healthy">Normal weight</div>', unsafe_allow_html=True)
        elif 25 <= bmi <= 29.9:
            st.markdown('<div class="metric-container warning">Overweight</div>', unsafe_allow_html=True)
        elif 30 <= bmi <= 34.9:
            st.markdown('<div class="metric-container danger">Obesity (Class 1)</div>', unsafe_allow_html=True)
        elif 35 <= bmi <= 39.9:
            st.markdown('<div class="metric-container danger">Obesity (Class 2)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container danger">Obesity (Class 3, Severe)</div>', unsafe_allow_html=True)

        blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)
        if blood_glucose <= 100:
            st.markdown('<div class="metric-container healthy">Normal (≤100 mg/dL)</div>', unsafe_allow_html=True)
        elif 101 <= blood_glucose <= 125:
            st.markdown('<div class="metric-container warning">Pre-diabetes (101–125 mg/dL)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container danger">Diabetic range (≥126 mg/dL)</div>', unsafe_allow_html=True)

        hba1c_level = st.slider("HbA1c Level (%)", 3.0, 15.0, 5.5, 0.1, format="%.1f")
        if hba1c_level < 5.7:
            st.markdown('<div class="metric-container healthy">Normal (<5.7%)</div>', unsafe_allow_html=True)
        elif 5.7 <= hba1c_level <= 6.4:
            st.markdown('<div class="metric-container warning">Pre-diabetes (5.7–6.4%)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container danger">Diabetes (≥6.5%)</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Check Risk", use_container_width=True)

    if submitted:
        model = load_model()
        gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
            gender, hypertension, heart_disease, smoking_history
        )
        input_data = np.array([[
            gender_val, age, hypertension_val, heart_disease_val,
            smoking_val, bmi, hba1c_level, blood_glucose
        ]])

        try:
            prediction = model.predict(input_data)[0]
            st.markdown("---")
            st.subheader("Prediction")
            if prediction == 0:
                st.success("No signs of diabetes detected.")
            else:
                st.warning("Possible risk of diabetes detected. Consult a healthcare professional for further evaluation.")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<small>
This tool is for educational use only and should not be used as a substitute for professional medical advice.  
Always consult a qualified healthcare professional for any medical concerns.
</small>
""", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
