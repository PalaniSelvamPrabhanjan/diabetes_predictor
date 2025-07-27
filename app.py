import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Configure page
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Title and description
st.title("🩺 Diabetes Risk Predictor")
st.markdown("""
This tool uses a machine learning model to estimate your risk of diabetes based on health indicators.  
**Please remember: This is not a medical diagnosis. Always consult a healthcare professional.**
""")

st.markdown("---")

# Load the model (with error handling)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.joblib')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'diabetes_model.joblib' not found. Please ensure the file is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

# Function to encode categorical variables (adjust based on your training encoding)
def encode_features(gender, hypertension, heart_disease, smoking_history):
    """
    Encode categorical features to match training data encoding.
    Adjust these mappings based on how your model was trained.
    """
    # Gender encoding
    gender_encoded = 1 if gender == "male" else 0
    
    # Hypertension encoding  
    hypertension_encoded = 1 if hypertension == "positive" else 0
    
    # Heart disease encoding
    heart_disease_encoded = 1 if heart_disease == "positive" else 0
    
    # Smoking history encoding (adjust based on your label encoding)
    smoking_map = {
        "No Info": 0,
        "Current": 1, 
        "Never": 2,
        "Past": 3
    }
    smoking_encoded = smoking_map[smoking_history]
    
    return gender_encoded, hypertension_encoded, heart_disease_encoded, smoking_encoded

# Create the prediction form
with st.form("diabetes_prediction_form"):
    st.subheader("Enter Your Health Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorical inputs
        gender = st.selectbox(
            "Gender",
            ["male", "female"],
            index=0
        )
        
        hypertension = st.selectbox(
            "Hypertension",
            ["negative", "positive"],
            index=0
        )
        
        heart_disease = st.selectbox(
            "Heart Disease", 
            ["negative", "positive"],
            index=0
        )
        
        smoking_history = st.selectbox(
            "Smoking History",
            ["No Info", "Current", "Never", "Past"],
            index=0
        )
    
    with col2:
        # Numeric inputs
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1
        )
        
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            format="%.1f"
        )
        
        # Sliders
        blood_glucose = st.slider(
            "Blood Glucose Level",
            min_value=50,
            max_value=300,
            value=100,
            step=1
        )
        
        hba1c_level = st.slider(
            "HbA1c Level",
            min_value=3.0,
            max_value=15.0,
            value=5.5,
            step=0.1,
            format="%.1f"
        )
    
    # Submit button
    submitted = st.form_submit_button("Check Risk", use_container_width=True)
    
    if submitted:
        # Load model
        model = load_model()
        
        # Encode categorical features
        gender_enc, hypertension_enc, heart_disease_enc, smoking_enc = encode_features(
            gender, hypertension, heart_disease, smoking_history
        )
        
        # Prepare input data (adjust column order to match your training data)
        input_data = np.array([[
            gender_enc,           # gender
            age,                  # age  
            hypertension_enc,     # hypertension
            heart_disease_enc,    # heart_disease
            smoking_enc,          # smoking_history
            bmi,                  # bmi
            hba1c_level,         # HbA1c_level
            blood_glucose        # blood_glucose_level
        ]])
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            if prediction == 0:
                # No diabetes
                st.success("❌ **No signs of diabetes detected**")
                st.balloons()
            else:
                # Diabetes detected
                st.warning("""
                ✅ **Possible signs of diabetes detected**
                
                Please note: This model is not always accurate.  
                Consult a healthcare professional for proper medical evaluation.
                """)
                
        except Exception as e:
            st.error(f"⚠️ Error making prediction: {str(e)}")

# Add footer information
st.markdown("---")
st.markdown("""
<small>
**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. 
Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.
</small>
""", unsafe_allow_html=True)

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)