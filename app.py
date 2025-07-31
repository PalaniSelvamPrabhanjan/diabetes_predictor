import streamlit as st
import joblib
import numpy as np
import base64
import time

# -----------------------------
# Page configuration
# -----------------------------
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
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            text-align: center;
            font-weight: 600;
        }}

        .green-box {{
            background-color: #e6f9ed;
            color: #065f46;
            border: 1px solid #34d399;
        }}

        .red-box {{
            background-color: #fde8e8;
            color: #991b1b;
            border: 1px solid #f87171;
        }}

        div.stButton {{
            padding-top: 2rem !important;
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
        }}

        /* Remove default range value blue background */
        span[data-testid="stTickLabel"] > div {{
            background: none !important;
            color: black !important;
        }}

        /* Slider track (the bar) */
        div.stSlider > div[data-baseweb="slider"] > div > div {{
            background: #a1daf8 !important;
            height: 6px;
            border-radius: 6px;
        }}

        /* Slider thumb (circle) */
        div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {{
            background-color: #a1daf8 !important;
            border: 2px solid #a1daf8 !important;
            width: 16px;
            height: 16px;
        }}

        /* Slider current value color */
        div[role="slider"] span {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("backgroundimage.jpg")

# -----------------------------
# Feature encoding
# -----------------------------
def encode_features(gender, hypertension, heart_disease, smoking_history):
    gender_val = 1 if gender == "male" else 0
    hypertension_val = 1 if hypertension == "positive" else 0
    heart_disease_val = 1 if heart_disease == "positive" else 0
    smoking_map = {"No Info": 0, "Current": 1, "Never": 2, "Past": 3}
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
    smoking_history = st.selectbox("Smoking History", ["No Info", "Current", "Never", "Past"])
with col4:
    heart_disease = st.selectbox("Heart Disease", ["negative", "positive"])

st.subheader("Health Metrics")
bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)
hba1c_level = st.slider("HbA1c Level (%) *", 3.0, 15.0, 5.5, 0.1)

# -----------------------------
# Prediction button and GIF
# -----------------------------
submitted = st.button("Check Risk", use_container_width=True)
gif_placeholder = st.empty()

if submitted:
    with open("loadingPage.gif", "rb") as f:
        base64_gif = base64.b64encode(f.read()).decode()
    gif_placeholder.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/gif;base64,{base64_gif}" width="60">
        </div>
        """,
        unsafe_allow_html=True
    )

    start_time = time.time()

    model = joblib.load("HGBCmodel.pkl")
    gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
        gender, hypertension, heart_disease, smoking_history
    )
    input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                            smoking_val, bmi, hba1c_level, blood_glucose]])
    prediction = model.predict(input_data)[0]

    elapsed = time.time() - start_time
    if elapsed < 3:
        time.sleep(3 - elapsed)

    gif_placeholder.empty()

    if prediction == 0:
        result_icon = "✅"
        result_text = "No Diabetes Risk Detected"
        explanation = "No signs of diabetes were detected based on the provided information."
        box_class = "green-box"
    else:
        result_icon = "⚠️"
        result_text = "Possible Diabetes Risk Detected"
        explanation = "Your results suggest a potential risk. Please consult a medical professional."
        box_class = "red-box"

    st.markdown(
        f"""
        <div class="result-box {box_class}">
            {result_icon} <strong>{result_text}</strong><br>
            <span style="font-size:0.9rem; font-weight:400;">{explanation}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<small>This tool is for rough prediction only. Always consult a medical professional.</small>", unsafe_allow_html=True)
