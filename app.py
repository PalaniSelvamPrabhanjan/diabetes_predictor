import streamlit as st
import joblib
import numpy as np
import base64
import time

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# Load Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

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
        div.stButton > button:first-child:active {{
            background-color: #003366 !important;
        }}
        div.stButton {{
            padding-top: 1rem;
        }}
        [data-testid="stTickBar"] {{
            background: none !important;
        }}
        [data-testid="stSliderTickBarMin"],
        [data-testid="stSliderTickBarMax"] {{
            color: #003366 !important;
            font-weight: 600;
        }}
        div[data-baseweb="slider"] .st-cv {{
            background: #a1daf8 !important;
        }}
        div[data-baseweb="slider"] .st-emotion-cache-1dj3ksd {{
            background-color: #a1daf8 !important;
            border: 2px solid #6cc3dd !important;
        }}
        div[data-testid="stSliderThumbValue"] {{
            color: #003366 !important;
        }}
        .lifestyle-title {{
            text-align: left;
            margin-top: 1rem;
            color: #991b1b !important;
        }}
        .disclaimer-box {{
            background-color: #fff9db;
            margin-top: 0.5rem;
            border: 1px solid #fcd34d;
            color: #92400e;
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.05rem;
            font-weight: 600;
        }}

        /* Hover effect */
        button[data-testid="stNumberInputStepDown"]:hover,
        button[data-testid="stNumberInputStepUp"]:hover {{
            background-color: #7ccbeb !important;
        }}

        /* Hover effect */
        button[data-testid="stNumberInputStepDown"]:active,
        button[data-testid="stNumberInputStepUp"]:active {{
            background-color: #7ccbeb !important;
        }}

        /* Icons inside the + and - buttons */
        button[data-testid="stNumberInputStepDown"] svg,
        button[data-testid="stNumberInputStepUp"] svg {{
            fill: white !important;
        }}

        /* Highlight selected number input value */
        input[type="number"]:focus {{
            border-color: #a1daf8 !important;
            box-shadow: 0 0 0 0.15rem rgba(161, 218, 248, 0.5) !important;
            color: #003366 !important;
        }}

        /* Override red border on selectbox active/focus state */
        div[role="combobox"]:focus-within {{
            border: none !important;
            box-shadow: 0 0 0 2px #a1daf8 !important;  
            outline: none !important;
        }}

        /* Remove red hover ring */
        div[role="combobox"]:hover {{
            border: none !important;
            box-shadow: 0 0 0 2px #a1daf8 !important; 
        }}

        /* Optional: Inner input field if needed */
        div[role="combobox"] input:focus {{
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
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
        "never": 0,
        "former Light Smoker": 1,
        "former Heavy Smoker": 2,
        "current": 3,
        "no Info": 4
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
        "no Info", "never", "former Light Smoker", "former Heavy Smoker", "current"
    ])
with col4:
    heart_disease = st.selectbox("Heart Disease", ["negative", "positive"])

st.subheader("Health Metrics")
blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)
hba1c_level = st.slider("HbA1c Level (%) *", 3.0, 15.0, 5.5, 0.1)

col5, col6 = st.columns(2)
with col5:
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
with col6:
    weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
height_m = height_cm / 100
bmi = round(weight_kg / (height_m ** 2), 2)
st.markdown(f"<p>Calculated BMI: {bmi}</p>", unsafe_allow_html=True)

# -----------------------------
# Submit Button and GIF
# -----------------------------
submitted = st.button("Check Risk", use_container_width=True)
gif_placeholder = st.empty()

if submitted:
    with open("loadingPage.gif", "rb") as f:
        base64_gif = base64.b64encode(f.read()).decode()
    gif_placeholder.markdown(
        f"""<div style="text-align:center;"><img src="data:image/gif;base64,{base64_gif}" width="60"></div>""",
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
        st.markdown("""
            <div class="result-box green-box">
                <i class="fas fa-circle-check"></i> <strong> No Diabetes Risk Detected</strong><br>
                <span>No signs of diabetes were detected based on the provided information.</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        notes = ""
        if blood_glucose > 200:
            notes += "<li>Blood Glucose Level is high (above 200 mg/dL)</li>"
        if hba1c_level > 6.5:
            notes += "<li>HbA1c Level is high (above 6.5%)</li>"

        st.markdown(f"""
            <div class="result-box red-box">
                <i class="fas fa-triangle-exclamation"></i> <strong> Possible Diabetes Risk Detected</strong><br>
                <span>Your results suggest a potential risk. Please consult a medical professional.</span>
                <ul style='text-align: left; margin-top: 1rem;'>{notes}</ul>
                <p class="lifestyle-title"><strong>Suggested lifestyle changes:</strong></p>
                <ul style='text-align: left; margin-bottom: 0;'>
                    <li>Switch to a diet rich in whole grains, lean proteins and vegetables.</li>
                    <li>Perform regular exercise.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Always-visible Disclaimer
# -----------------------------
st.markdown("""
    <div class="disclaimer-box">
        <i class="fas fa-exclamation-circle"></i> <strong> Medical Disclaimer</strong><br>
        <span>This tool is for rough prediction only. Always consult a medical professional for a confirmed diagnosis.</span>
    </div>
""", unsafe_allow_html=True)
