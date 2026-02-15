import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from predict import load_model, predict_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ScrapNet",
    page_icon="🌿",
    layout="wide",
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌿 ScrapNet")
    st.markdown("AI-powered waste classification")
    st.divider()

    st.markdown("### Model Info")
    st.write("Architecture: EfficientNet-B0")
    st.write("Framework: PyTorch")
    st.write("Device: CPU")

    st.divider()

    show_top3 = st.toggle("Show Top 3 Predictions", value=True)
    show_probs = st.toggle("Show Probability Chart", value=True)

    st.divider()
    st.caption("Built with Streamlit")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def get_model():
    return load_model(device="cpu")

model, classes = get_model()

# ---------------- HEADER ----------------
st.title("🌿 Waste Classification System")
st.caption("Upload a waste image and let the AI predict the category.")

st.divider()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader(
        "Supported: JPG, JPEG, PNG, WEBP",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

with col2:
    if uploaded:
        label, conf, probs = predict_image(model, classes, img, device="cpu")

        st.subheader("♻️ Prediction")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric("Predicted Class", label)

        with metric_col2:
            st.metric("Confidence", f"{conf*100:.2f}%")

        if show_top3:
            st.subheader("Top Predictions")

            probs = np.array(probs)
            top_idx = np.argsort(probs)[::-1][:3]

            df = pd.DataFrame({
                "Class": [classes[i] for i in top_idx],
                "Probability": [float(probs[i]) for i in top_idx]
            })

            st.dataframe(df, use_container_width=True)

        if show_probs:
            st.subheader("Probability Distribution")
            st.bar_chart(pd.Series(probs, index=classes))

    else:
        st.info("Upload an image to see prediction results.")

st.divider()
st.caption("© 2026 ScrapNet | Sustainable AI")
