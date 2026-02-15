import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from predict import load_model, predict_image

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="ScrapNet • Eco Waste Classifier",
    page_icon="🌿",
    layout="centered",
)

# ---------------- ECO STYLING ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    padding-top: 1rem;
}

.hero {
    text-align:center;
    padding: 2rem 1rem;
    border-radius: 22px;
    background: linear-gradient(135deg, #0f5132, #198754);
    color: white;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 2.4rem;
    margin-bottom: 0.5rem;
}

.sub {
    opacity: 0.9;
}

.card {
    background: rgba(25, 135, 84, 0.06);
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(25,135,84,0.25);
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}

.badge {
    display:inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    background: #d1e7dd;
    color: #0f5132;
    font-size: 0.85rem;
    font-weight: 600;
}

.metric-green {
    font-size: 1.3rem;
    font-weight: 600;
    color: #198754;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def get_model():
    return load_model(device="cpu")

model, classes = get_model()

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <h1>🌿 ScrapNet</h1>
    <div class="sub">AI-Powered Waste Classification using EfficientNet-B0</div>
    <br>
    <span class="badge">Sustainable AI • Transfer Learning</span>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- Upload Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Waste Image")
uploaded = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG, WEBP",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------- Prediction ----------------
if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        label, conf, probs = predict_image(model, classes, img, device="cpu")

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ♻️ Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-green'>Class:</div>", unsafe_allow_html=True)
            st.markdown(f"## {label}")

        with col2:
            st.markdown(f"<div class='metric-green'>Confidence:</div>", unsafe_allow_html=True)
            st.markdown(f"## {conf*100:.2f}%")

        probs = np.array(probs)
        top_idx = np.argsort(probs)[::-1][:3]

        df = pd.DataFrame({
            "Class": [classes[i] for i in top_idx],
            "Probability": [float(probs[i]) for i in top_idx]
        })

        st.write("")
        st.markdown("### 🌱 Top 3 Predictions")
        st.bar_chart(df.set_index("Class")["Probability"])

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception:
        st.error("⚠️ Could not process image. Try a clearer or smaller file.")

else:
    st.info("🌍 Upload an image to see AI classify the waste category.")
