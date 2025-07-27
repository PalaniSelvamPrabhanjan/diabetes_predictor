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
# Background & Custom CSS (No button overrides)
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
        /* ✅ Result Card */
        .result-box {{
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.1rem;
            margin-top: 1rem;
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

    # ✅ HbA1c Title + Caption on Same Row (Cleaned)
    col_hba1c1, col_hba1c2 = st.columns([1.2, 2.5])
    with col_hba1c1:
        st.markdown("**HbA1c Level (%)**")
    with col_hba1c2:
        st.caption("Average sugar (2-3 months). ≥6.5% = diabetic.")

    hba1c_level = st.slider("", 3.0, 15.0, 5.5, 0.1)  # Empty label

    # ✅ Other Sliders
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
    blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)

    # ✅ Padding Above Button
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Check Risk", use_container_width=True)

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:
    with st.spinner("🔄 Checking your diabetes risk..."):
        model = load_model()
        gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
            gender, hypertension, heart_disease, smoking_history
        )
        input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                                smoking_val, bmi, hba1c_level, blood_glucose]])
        prediction = model.predict(input_data)[0]

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

    st.markdown("---")
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
st.markdown("<small>This tool is for educational use only. Always consult a medical professional.</small>", unsafe_allow_html=True)
