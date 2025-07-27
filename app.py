import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

def add_background(image_file):
    with open(image_file, "rb") as img:
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
        .metric-container {{
            background-color: rgba(255,255,255,0.9);
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.4rem 0;
            border-left: 4px solid;
        }}
        .healthy {{ border-left-color: #22c55e; }}
        .warning {{ border-left-color: #f59e0b; }}
        .danger {{ border-left-color: #ef4444; }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background("backgroundImg.jpg")

def slider_color(value, healthy, warning):
    if healthy[0] <= value <= healthy[1]:
        return "#22c55e"
    elif warning[0] <= value <= warning[1]:
        return "#f59e0b"
    return "#ef4444"

def update_slider_style(index, color):
    st.markdown(
        f"""
        <style>
        div.row-widget.stSlider:nth-of-type({index}) [data-baseweb="slider"] > div > div {{
            background: linear-gradient(to right, {color} 0%, {color} 100%) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    try:
        return joblib.load("HGBCmodel.pkl")
    except FileNotFoundError:
        st.error("Model file 'HGBCmodel.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def encode_inputs(gender, hypertension, heart_disease, smoking):
    gender_val = 1 if gender == "male" else 0
    hypertension_val = 1 if hypertension == "positive" else 0
    heart_disease_val = 1 if heart_disease == "positive" else 0
    smoking_map = {"No Info": 0, "Current": 1, "Never": 2, "Past": 3}
    return gender_val, hypertension_val, heart_disease_val, smoking_map[smoking]

st.title("Diabetes Risk Predictor")
st.markdown(
    "Provide your health details to estimate the risk of diabetes. "
    "**This tool is for informational purposes and not a medical diagnosis.**"
)
st.markdown("---")

with st.form("diabetes_form"):
    st.subheader("Your Health Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        hypertension = st.selectbox("Hypertension", ["negative", "positive"])
        heart_disease = st.selectbox("Heart Disease", ["negative", "positive"])
        smoking = st.selectbox("Smoking History", ["No Info", "Current", "Never", "Past"])

    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
        bmi_color = slider_color(bmi, (18.5, 24.9), (25, 29.9))
        update_slider_style(1, bmi_color)
        if bmi < 18.5:
            st.markdown('<div class="metric-container warning">Underweight</div>', unsafe_allow_html=True)
        elif 18.5 <= bmi <= 24.9:
            st.markdown('<div class="metric-container healthy">Normal weight</div>', unsafe_allow_html=True)
        elif 25 <= bmi <= 29.9:
            st.markdown('<div class="metric-container warning">Overweight</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container danger">Obesity</div>', unsafe_allow_html=True)

        glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 100)
        glucose_color = slider_color(glucose, (0, 100), (101, 125))
        update_slider_style(2, glucose_color)
        if glucose <= 100:
            st.markdown('<div class="metric-container healthy">Normal (≤100 mg/dL)</div>', unsafe_allow_html=True)
        elif 101 <= glucose <= 125:
            st.markdown('<div class="metric-container warning">Pre-diabetes (101–125 mg/dL)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container danger">Diabetic range (≥126 mg/dL)</div>', unsafe_allow_html=True)

        hba1c = st.slider("HbA1c Level (%)", 3.0, 15.0, 5.5, 0.1)
        hba1c_color = slider_color(hba1c, (0, 5.6), (5.7, 6.4))
        update_slider_style(3, hba1c_color)
        if hba1c < 5.7:
            st.markdown('<div class="metric-container healthy">Normal (<5.7%)</div>', unsafe_allow_html=True)
        elif 5.7 <= hba1c <= 6.4:
            st.markdown('<div class="metric-container warning">Pre-diabetes (5.7–6.4%)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container danger">Diabetes (≥6.5%)</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Check Risk", use_container_width=True)

    if submitted:
        with st.spinner("Processing..."):
            model = load_model()
            gender_val, hypertension_val, heart_disease_val, smoking_val = encode_inputs(
                gender, hypertension, heart_disease, smoking
            )
            data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                              smoking_val, bmi, hba1c, glucose]])
            try:
                result = model.predict(data)[0]
                st.markdown("---")
                st.subheader("Prediction")
                if result == 0:
                    st.success("No signs of diabetes detected.")
                else:
                    st.warning("Possible risk of diabetes detected. Consult a healthcare professional.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown(
    "<small>This tool is for educational use only. Always seek professional medical advice.</small>",
    unsafe_allow_html=True
)
