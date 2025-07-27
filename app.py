import streamlit as st
import joblib
import numpy as np
import base64
import time

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="centered"
)

# -----------------------------
# Background & CSS
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
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            max-width: 800px;
            margin: 2rem auto;
        }}
        .stMarkdown, label, p, h1, h2, h3, h4, h5, h6 {{
            color: black !important;
        }}
        .stButton>button {{
            background: #2563eb !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.6rem 1rem !important;
            font-weight: bold !important;
        }}
        .stButton>button:hover {{
            background: #1d4ed8 !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("backgroundimage.jpg")

# -----------------------------
# Encode Loading GIF
# -----------------------------
def get_gif_base64(gif_path):
    with open(gif_path, "rb") as gif:
        return base64.b64encode(gif.read()).decode()

loading_gif = get_gif_base64("loadingPage.gif")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("HGBCmodel.pkl")

def encode_features(gender, hypertension, heart_disease, smoking_history):
    gender_val = 1 if gender == "male" else 0
    hypertension_val = 1 if hypertension == "positive" else 0
    heart_disease_val = 1 if heart_disease == "positive" else 0
    smoking_map = {"No Info": 0, "Current": 1, "Never": 2, "Past": 3}
    return gender_val, hypertension_val, heart_disease_val, smoking_map[smoking_history]

# -----------------------------
# Page Title
# -----------------------------
st.markdown("<h1 style='text-align:center;'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Estimate your diabetes risk based on health indicators.<br><b>This is not medical advice.</b></p>", unsafe_allow_html=True)

# -----------------------------
# Session State for Loading and Result
# -----------------------------
if "loading" not in st.session_state:
    st.session_state.loading = False
if "result" not in st.session_state:
    st.session_state.result = None

# -----------------------------
# Input Form
# -----------------------------
with st.form("diabetes_form"):
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
    st.caption("* HbA1c reflects average blood sugar over the past 2-3 months. ≥6.5% is considered diabetic.")

    submitted = st.form_submit_button("Check Risk", use_container_width=True)

# -----------------------------
# Prediction Logic with Loading GIF
# -----------------------------
if submitted:
    st.session_state.loading = True
    st.session_state.result = None
    st.experimental_rerun()

if st.session_state.loading:
    # Show Centered Loading GIF
    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 30px;">
            <img src="data:image/gif;base64,{loading_gif}" width="120">
            <p style="color:#374151; font-size:16px;">Analyzing your health data...</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Simulate processing delay
    time.sleep(1.5)

    # Run Prediction
    model = load_model()
    gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
        gender, hypertension, heart_disease, smoking_history
    )
    input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                            smoking_val, bmi, hba1c_level, blood_glucose]])
    prediction = model.predict(input_data)[0]

    st.session_state.result = prediction
    st.session_state.loading = False
    st.experimental_rerun()

# -----------------------------
# Show Result in a Clean Card
# -----------------------------
if st.session_state.result is not None:
    if st.session_state.result == 0:
        st.markdown(
            """
            <div style="
                background-color: #ecfdf5;
                border: 1px solid #34d399;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                text-align: center;
                margin-top: 1rem;
            ">
                <h3 style="color: #065f46; margin: 0;">✅ No Diabetes Risk Detected.</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background-color: #fef2f2;
                border: 1px solid #f87171;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                text-align: center;
                margin-top: 1rem;
            ">
                <h3 style="color: #991b1b; margin: 0;">⚠️ Possible Diabetes Risk Detected. Please consult a medical professional.</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<small>This tool is for educational use only. Always consult a medical professional.</small>", unsafe_allow_html=True)