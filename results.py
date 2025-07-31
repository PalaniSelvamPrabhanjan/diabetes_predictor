import streamlit as st
import base64

st.set_page_config(page_title="Result", layout="centered")

# Optional: set same background so both pages match
def set_background(image_path: str):
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
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("backgroundimage.jpg")

payload = st.session_state.get("result_payload")

# If this page was opened directly, go back to main app
if payload is None:
    st.info("No result found. Returning to the main page.")
    st.switch_page("app.py")

pred = payload["prediction"]
box_class = "green-box" if pred == 0 else "red-box"
title = "No Diabetes Risk Detected" if pred == 0 else "Possible Diabetes Risk Detected"
desc = (
    "No signs of diabetes were detected based on the provided information."
    if pred == 0 else
    "Your results suggest a potential risk. Please consult a medical professional."
)

st.markdown("<h1 style='text-align:center;'>Result</h1>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="result-box {box_class}">
        <strong>{title}</strong><br>
        <span style="font-size:0.9rem; font-weight:400;">{desc}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()
with st.expander("View inputs"):
    st.json(payload["inputs"])

# Back button to main page
if st.button("← Back"):
    st.switch_page("app.py")

st.markdown("---")
st.markdown("<small>This tool is for rough prediction only. Always consult a medical professional.</small>", unsafe_allow_html=True)
