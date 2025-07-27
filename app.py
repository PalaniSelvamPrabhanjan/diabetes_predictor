import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="centered"
)

# -----------------------------
# Custom CSS (Background + Cards)
# -----------------------------
st.markdown("""
<style>
/* Full-page background image */
.stApp {
    background: url("backgroundimage.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Card-like white container */
.reportview-container .main .block-container {
    padding: 2rem 2rem 2rem 2rem;
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    max-width: 800px;
    margin: auto;
}

/* Titles */
h1, h2, h3 {
    color: #2563eb; /* Blue tone */
    font-weight: bold;
}

/* Subheaders */
.subheader {
    color: #374151;
    font-size: 1rem;
    font-weight: 400;
}

/* Warning, success, error boxes */
div.stAlert {
    border-radius: 8px;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #10b981);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
    font-size: 1rem;
}
.stButton>button:hover {
    opacity: 0.9;
}

/* Metric boxes */
.metric-container {
    background-color: rgba(255,255,255,0.9);
    padding: 0.8rem;
    border-radius: 8px;
    margin: 0.4rem 0;
    border-left: 4px solid;
}
.healthy { border-left-color: #22c55e; }
.warning { border-left-color: #f59e0b; }
.danger { border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load the Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("HGBCmodel.pkl")
    except FileNotFoundError:
        st.error("Model file 'HGBCmodel.pkl' not found. Make sure it is in the same folder as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# -----------------------------
# Encode Features
# -----------------------------
def encode_features(gender, hypertension, heart_disease, smoking_history):
    gender_val = 1 if gender == "Male" else 0
    hypertension_val = 1 if hypertension == "Positive" else 0
    heart_disease_val = 1 if heart_disease == "Positive" else 0
    smoking_map = {"No Info": 0, "Current": 1, "Never": 2, "Past": 3}
    smoking_val = smoking_map[smoking_history]
    return gender_val, hypertension_val, heart_disease_val, smoking_val

# -----------------------------
# Title & Disclaimer
# -----------------------------
st.title("Diabetes Risk Predictor")
st.markdown("""
This tool uses machine learning to estimate your diabetes risk based on health indicators.  
<span style="color:red; font-weight:bold;">⚠ This is not a medical diagnosis. Always consult a healthcare professional.</span>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Input Form (Card Layout)
# -----------------------------
with st.form("diabetes_prediction_form"):
    st.subheader("❤️ Health Assessment Form")
    st.markdown("Please provide accurate information for better prediction accuracy.")

    st.markdown("### DEMOGRAPHICS")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender *", ["Male", "Female"])
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    st.markdown("### MEDICAL HISTORY")
    col3, col4 = st.columns(2)
    with col3:
        hypertension = st.selectbox("Hypertension *", ["Negative", "Positive"])
    with col4:
        heart_disease = st.selectbox("Heart Disease *", ["Negative", "Positive"])

    smoking_history = st.selectbox("Smoking History *", ["No Info", "Current", "Never", "Past"])

    st.markdown("### HEALTH METRICS")
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1, format="%.1f")
    blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)
    hba1c_level = st.slider("HbA1c Level (%)", 3.0, 15.0, 5.5, 0.1, format="%.1f")

    submitted = st.form_submit_button("Check Diabetes Risk")

    if submitted:
        model = load_model()
        gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
            gender, hypertension, heart_disease, smoking_history
        )

        input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                                smoking_val, bmi, hba1c_level, blood_glucose]])

        try:
            prediction = model.predict(input_data)[0]
            st.markdown("---")
            if prediction == 0:
                st.success("✅ No signs of diabetes detected.")
            else:
                st.error("❌ Possible Diabetes Risk Detected")
                st.warning("Possible signs of diabetes detected. Consult a healthcare professional for proper evaluation.")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="background-color:#fff3cd; padding:10px; border-radius:5px;">
<b>⚠ Medical Disclaimer</b><br>
This tool is for educational purposes only and should not replace professional medical advice.  
Always consult with a healthcare provider for accurate diagnosis and treatment.
</div>
""", unsafe_allow_html=True)
