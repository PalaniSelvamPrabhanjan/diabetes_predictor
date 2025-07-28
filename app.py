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
def set_background(image_path, show_form=True):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    container_style = """
        .block-container {
            background-color: rgba(255,255,255,0.97);
            padding: 2rem 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            max-width: 800px;
            margin: 2rem auto;
        }
    """ if show_form else ""  # hide white overlay when loading

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        {container_style}
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
        div.stButton > button:first-child span {{
            color: white !important;
        }}
        div.stButton > button:first-child {{
            background-color: #1E88E5 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #1565C0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# UI and Logic
# -----------------------------
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if not st.session_state.predict_clicked:
    set_background("backgroundimage.jpg", show_form=True)

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

    submitted = st.button("Check Risk", use_container_width=True)
    if submitted:
        st.session_state.predict_clicked = True
        st.session_state.inputs = (gender, age, hypertension, heart_disease, smoking_history, bmi, blood_glucose, hba1c_level)
        st.rerun()

else:
    set_background("backgroundimage.jpg", show_form=False)

    gif_placeholder = st.empty()
    with open("loadingPage.gif", "rb") as f:
        gif = base64.b64encode(f.read()).decode()
    gif_placeholder.markdown(
        f"""
        <div style="display:flex;justify-content:center;align-items:center;height:100vh;">
            <img src="data:image/gif;base64,{gif}" width="120">
        </div>
        """,
        unsafe_allow_html=True
    )

    start_time = time.time()
    model = joblib.load("HGBCmodel.pkl")
    gender, age, hypertension, heart_disease, smoking_history, bmi, blood_glucose, hba1c_level = st.session_state.inputs

    def encode_features(gender, hypertension, heart_disease, smoking_history):
        gender_val = 1 if gender == "male" else 0
        hypertension_val = 1 if hypertension == "positive" else 0
        heart_disease_val = 1 if heart_disease == "positive" else 0
        smoking_map = {"No Info": 0, "Current": 1, "Never": 2, "Past": 3}
        return gender_val, hypertension_val, heart_disease_val, smoking_map[smoking_history]

    gender_val, hypertension_val, heart_disease_val, smoking_val = encode_features(gender, hypertension, heart_disease, smoking_history)
    input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val, smoking_val, bmi, hba1c_level, blood_glucose]])
    prediction = model.predict(input_data)[0]

    elapsed = time.time() - start_time
    if elapsed < 3:
        time.sleep(3 - elapsed)

    gif_placeholder.empty()
    st.session_state.predict_clicked = False
    st.rerun()
