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
# Background Image
# -----------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .card {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("backgroundimage.jpg")

# -----------------------------
# Loading Overlay
# -----------------------------
st.markdown("""
<style>
#loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.85);
    z-index: 9999;
    display: flex;
    justify-content: center;
    align-items: center;
    visibility: hidden;
}
</style>

<div id="loading-overlay">
    <img src="data:image/gif;base64,{}" width="180">
</div>
""".format(
    base64.b64encode(open("loadingPage.gif", "rb").read()).decode()
), unsafe_allow_html=True)

# -----------------------------
# Helper Functions
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
# Title and Description
# -----------------------------
st.markdown("<h1 style='text-align:center;'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Estimate your diabetes risk based on health indicators.<br><b style='color:red;'>This is not medical advice.</b></p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Input Form
# -----------------------------
with st.form("diabetes_form"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    st.subheader("Medical History")
    col3, col4 = st.columns(2)
    with col3:
        hypertension = st.selectbox("Hypertension", ["negative", "positive"])
        smoking_history = st.selectbox("Smoking History", ["No Info", "Current", "Never", "Past"])
    with col4:
        heart_disease = st.selectbox("Heart Disease", ["negative", "positive"])

    st.subheader("Health Metrics")
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1, format="%.1f")
    blood_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100, 1)
    hba1c_level = st.slider("HbA1c Level (%) *", 3.0, 15.0, 5.5, 0.1, format="%.1f")
    st.caption("* HbA1c reflects average blood sugar over the past 2-3 months. A level ≥6.5% is diabetic.")

    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Check Risk", use_container_width=True)

# -----------------------------
# Prediction and Result Modal
# -----------------------------
if submitted:
    # Show loading overlay
    st.markdown("""
        <script>
        var overlay = parent.document.querySelector('#loading-overlay');
        overlay.style.visibility = 'visible';
        </script>
    """, unsafe_allow_html=True)

    model = load_model()
    gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(
        gender, hypertension, heart_disease, smoking_history
    )
    input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                            smoking_val, bmi, hba1c_level, blood_glucose]])
    prediction = model.predict(input_data)[0]

    # Hide loading overlay
    st.markdown("""
        <script>
        var overlay = parent.document.querySelector('#loading-overlay');
        overlay.style.visibility = 'hidden';
        </script>
    """, unsafe_allow_html=True)

    # Result Card
    if prediction == 0:
        st.markdown(
            """
            <div style="background:#d1fae5; padding:1rem; border-radius:10px; border:2px solid #10b981;">
            <h3 style="color:#065f46; text-align:center;">No Diabetes Risk Detected</h3>
            <p style="text-align:center;">Your indicators do not suggest diabetes risk.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="background:#fee2e2; padding:1rem; border-radius:10px; border:2px solid #b91c1c;">
            <h3 style="color:#7f1d1d; text-align:center;">Possible Diabetes Risk Detected</h3>
            <p style="text-align:center;">Consult a healthcare professional for further evaluation.</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<small>This tool is for educational use only. Always seek professional medical advice.</small>", unsafe_allow_html=True)
