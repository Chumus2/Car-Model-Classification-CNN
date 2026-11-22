import os
import requests
from PIL import Image
import streamlit as st
from cv.src import configs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(BASE_DIR, 'app-icon.png')
page_icon = Image.open(icon_path) if os.path.exists(icon_path) else "🚗"

st.set_page_config(
    page_title="Car AI Classifier",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("Endpoint", value=configs.API_URL)
    st.caption("ResNet18 x FastAPI + Streamlit")

st.title("Car AI Classifier")
st.write("Upload a car image and click Predict to classify the  model.")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    uploaded_file = st.file_uploader("Choose an image", type=["jpeg", "png", "jpg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        run_pred = st.button("Predict", type="primary", use_container_width=True)
    else:
        run_pred = False

with col_right:
    st.subheader("Result")

    if uploaded_file is None:
        st.info("Upload an image first.")
    else:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/jpeg"
            )
        }

        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(configs.API_URL, files=files, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
                st.stop()

        st.success(f"**{data['pred_label']}**")
        st.metric("Confidence", f"{float(data['confidence']):.2f}%")