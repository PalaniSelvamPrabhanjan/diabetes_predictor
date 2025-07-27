import streamlit as st
import joblib
import numpy as np
import base64

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="centered"
)

# -----------------------------
# Background Image & Styles
# -----------------------------
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main-card {{
            background-color: rgba(255,255,255,0.95);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 2rem auto;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: bold;
            color: black;
            margin-top: 1rem;
        }}
        .stMarkdown, label, p, h1, h2, h3, h4, h5, h6 {{
            color: black !important;
        }}
        .stButton>button {{
            background: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background: #1d4ed8;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("backgroundimage.jpg")

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
# Input Form (inside white card)
# -----------------------------
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    with st.form("diabetes_form"):
        st.markdown("<div class='section-title'>Demographics</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["male", "female"])
        with col2:
            age = st.slider("Age", 0, 120, 30, 1)

        st.markdown("<div class='section-title'>Medical History</div>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            hypertension = st.selectbox("Hypertension", ["negative", "positive"])
            smoking_history = st.selectbox("Smoking History", ["No Info", "Current", "Never", "Past"])
        with col4:
            heart_disease = st.selectbox("Heart Disease", ["negative", "positive"])

        st.markdown("<div class='section-title'>Health Metrics</div>", unsafe_allow_html=True)
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
        blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)
        hba1c_level = st.slider("HbA1c Level (%) *", 3.0, 15.0, 5.5, 0.1)
        st.caption("* HbA1c reflects average blood sugar over the past 2-3 months. ≥6.5% is considered diabetic.")

        submitted = st.form_submit_button("Check Risk", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction & Pop-Up Modal
# -----------------------------
if submitted:
    model = load_model()
    gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
        gender, hypertension, heart_disease, smoking_history
    )
    input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                            smoking_val, bmi, hba1c_level, blood_glucose]])
    prediction = model.predict(input_data)[0]

    result_text = (
        "✅ No Diabetes Risk Detected." if prediction == 0
        else "⚠️ Possible Diabetes Risk Detected. Please consult a medical professional."
    )

    st.markdown(
        f"""
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.4);
            z-index: 9998;
        "></div>

        <div style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.2);
            z-index: 9999;
            text-align: center;
        ">
            <h3 style="color: {'green' if prediction == 0 else 'red'};">{result_text}</h3>
            <button onclick="window.location.reload()" style="
                margin-top: 1rem;
                background: #2563eb;
                color: white;
                border: none;
                padding: 0.6rem 1rem;
                border-radius: 8px;
                cursor: pointer;
            ">Close</button>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<small>This tool is for educational use only. Always consult a medical professional.</small>", unsafe_allow_html=True)
