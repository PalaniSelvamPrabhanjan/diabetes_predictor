# -----------------------------
# Input Form (inside white card)
# -----------------------------
with st.container():
    st.markdown(
        """
        <div class='main-card'>
        """,
        unsafe_allow_html=True
    )

    # ✅ All Streamlit elements remain inside this container
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
