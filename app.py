import streamlit as st
import joblib
import numpy as np
import base64
import time

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# -----------------------------
# Background & Custom CSS
# -----------------------------
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

        .stApp {{
            background: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(255,255,255,0.97);
            padding: 2rem 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            max-width: 800px;
            margin: 2rem auto;
        }}
        .stMarkdown, label, p, h1, h2, h3, h4, h5, h6 {{
            color: black !important;
        }}
        .result-box {{
            padding: 1.5rem;
            border-radius: 12px;
            font-size: 1.05rem;
            margin-top: 1rem;
            font-weight: 500;
        }}
        .green-box {{
            background-color: #e6f9ed;
            color: #065f46;
            border: 2px solid #34d399;
        }}
        .red-box {{
            background-color: #fde8e8;
            color: #991b1b;
            border: 2px solid #f87171;
        }}
        .result-box ul {{
            margin-top: 0.4rem;
            padding-left: 1.2rem;
        }}
        .result-box li {{
            margin-bottom: 0.3rem;
        }}
        .disclaimer-box {{
            background-color: #fef9c3;
            border: 2px solid #facc15;
            border-radius: 12px;
            padding: 1.2rem;
            margin-top: 1rem;
            font-size: 1rem;
            color: #78350f;
        }}
        .disclaimer-box i {{
            margin-right: 0.5rem;
        }}
        div.stButton > button:first-child {{
            background-color: #a1daf8 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            margin-top: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("backgroundimage.jpg")

# -----------------------------
# Encode features
# -----------------------------
def encode_features(gender, hypertension, heart_disease, smoking_history):
    gender_val = 1 if gender == "male" else 0
    hypertension_val = 1 if hypertension == "positive" else 0
    heart_disease_val = 1 if heart_disease == "positive" else 0
    smoking_map = {
        "No Info": 0,
        "Never": 1,
        "Former Low Risk": 2,
        "Former High Risk": 3,
        "Current": 4
    }
    return gender_val, hypertension_val, heart_disease_val, smoking_map[smoking_history]

# -----------------------------
# UI
# -----------------------------
st.markdown("<h1 style='text-align:center;'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Estimate your diabetes risk based on health indicators.<br><b>This is not medical advice.</b></p>", unsafe_allow_html=True)

st.subheader("Demographics")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["male", "female"])
with col2:
    age = st.slider("Age", 0, 120, 30, 1)

st.subheader("Medical History")
col3, col4 = st.columns(2)
with col3:
    hypertension = st.selectbox("Hypertension", ["negative", "positive"])
    smoking_history = st.selectbox("Smoking History", [
        "No Info", "Never", "Former Low Risk", "Former High Risk", "Current"
    ])
with col4:
    heart_disease = st.selectbox("Heart Disease", ["negative", "positive"])

st.subheader("Health Metrics")
bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)
hba1c_level = st.slider("HbA1c Level (%) *", 3.0, 15.0, 5.5, 0.1)

# -----------------------------
# Button + Prediction
# -----------------------------
submitted = st.button("Check Risk", use_container_width=True)

if submitted:
    start_time = time.time()

    model = joblib.load("HGBCmodel.pkl")
    gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
        gender, hypertension, heart_disease, smoking_history
    )
    input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                            smoking_val, bmi, hba1c_level, blood_glucose]])
    prediction = model.predict(input_data)[0]

    elapsed = time.time() - start_time
    if elapsed < 2:
        time.sleep(2 - elapsed)

    if prediction == 0:
        st.markdown(
            f"""
            <div class="result-box green-box">
                <i class="fa-solid fa-circle-check fa-lg"></i> <strong>No Diabetes Risk Detected</strong><br>
                <span>No signs of diabetes detected based on the provided information.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        extra_messages = []
        if blood_glucose > 200:
            extra_messages.append("<li>High Blood Glucose</li>")
        if hba1c_level > 6.5:
            extra_messages.append("<li>HbA1c Level is high (above 6.5%)</li>")

        extra_html = f"<ul>{''.join(extra_messages)}</ul>" if extra_messages else ""

        st.markdown(
            f"""
            <div class="result-box red-box">
                <i class="fa-solid fa-triangle-exclamation fa-lg"></i> <strong>Possible Diabetes Risk Detected</strong><br>
                <span style="font-size: 1rem;">Your results suggest a potential risk. Please consult a medical professional.</span>
                {extra_html}
                <strong>Suggested lifestyle changes:</strong>
                <ul>
                    <li>Switch to a diet rich in whole grains, lean proteins and vegetables.</li>
                    <li>Perform regular exercise.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Always show disclaimer
# -----------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        <i class="fa-solid fa-circle-exclamation fa-lg"></i>
        <strong>Medical Disclaimer</strong><br>
        This tool is for rough prediction only. Always consult a medical professional for a confirmed diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)
