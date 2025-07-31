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
        .result-box {{
            padding: 1.2rem;
            border-radius: 12px;
            font-size: 1.1rem;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
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
        .yellow-box {{
            background-color: #fffbea;
            color: #92400e;
            border: 1px solid #fcd34d;
        }}
        .result-box ul {{
            padding-top: 0.3rem;
            margin-bottom: 0.5rem;
        }}
        .result-box li {{
            padding-top: 0.2rem;
        }}

        /* Remove caps above min/max */
        [data-testid="stTickBar"] {{
            background: none !important;
        }}

        /* Set all slider label values (tick labels) to black */
        span[data-testid="stTickLabel"] {{
            color: black !important;
            background: transparent !important;
        }}

        /* Optional: Ensure slider value above thumb is black */
        div[role="slider"] > div > span {{
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
    smoking_map = {
        "No Info": 0,
        "Never": 1,
        "Former Low Risk": 2,
        "Former High Risk": 3,
        "Current": 4
    }
    return gender_val, hypertension_val, heart_disease_val, smoking_map[smoking_history]

# -----------------------------
# UI Layout
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
# Prediction Button + Placeholder
# -----------------------------
st.markdown("<div style='padding-top: 1rem;'></div>", unsafe_allow_html=True)
submitted = st.button("Check Risk", use_container_width=True)
gif_placeholder = st.empty()

if submitted:
    with open("loadingPage.gif", "rb") as f:
        base64_gif = base64.b64encode(f.read()).decode()
    gif_placeholder.markdown(
        f"""<div style="text-align:center;"><img src="data:image/gif;base64,{base64_gif}" width="60"></div>""",
        unsafe_allow_html=True
    )

    start = time.time()
    model = joblib.load("HGBCmodel.pkl")

    g_val, h_val, hd_val, smoke_val = encode_features(gender, hypertension, heart_disease, smoking_history)
    X_input = np.array([[g_val, age, h_val, hd_val, smoke_val, bmi, hba1c_level, blood_glucose]])
    prediction = model.predict(X_input)[0]
    time.sleep(max(0, 3 - (time.time() - start)))
    gif_placeholder.empty()

    # Result box
    if prediction == 0:
        st.markdown(
            """
            <div class="result-box green-box">
                <p><i class="fas fa-check-circle"></i> <strong>No Diabetes Risk Detected</strong></p>
                <p style="margin-left:1.8rem;">No signs of diabetes detected based on the provided information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        extra_msgs = []
        if blood_glucose > 200:
            extra_msgs.append("<li>High Blood Glucose</li>")
        if hba1c_level > 6.5:
            extra_msgs.append("<li>HbA1c Level is high (above 6.5%)</li>")

        st.markdown(
            f"""
            <div class="result-box red-box">
                <p><i class="fas fa-exclamation-triangle"></i> <strong>Possible Diabetes Risk Detected</strong></p>
                <p><strong>Your results suggest a potential risk. Please consult a medical professional.</strong></p>
                {"<ul>" + "".join(extra_msgs) + "</ul>" if extra_msgs else ""}
                <p><strong>Suggested lifestyle changes:</strong></p>
                <ul>
                    <li>Switch to a diet rich in whole grains, lean proteins and vegetables.</li>
                    <li>Perform regular exercise.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Disclaimer Always Visible
# -----------------------------
st.markdown(
    """
    <div class="result-box yellow-box" style="margin-top: 0.5rem;">
        <p><i class="fas fa-exclamation-circle"></i> <strong>Medical Disclaimer</strong></p>
        <p>This tool is for rough prediction only. Always consult a medical professional for a confirmed diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)
