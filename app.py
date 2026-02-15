import os
import streamlit as st
from PIL import Image
import numpy as np
from predict import load_model, predict_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

def artifact_path(filename: str) -> str:
    return os.path.join(ARTIFACTS_DIR, filename)

def show_artifact_image(filename: str, caption: str | None = None):
    path = artifact_path(filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing artifact: `{filename}` (expected at `{path}`)")
        
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ScrapNet | Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# ---------------- ECO THEME CSS ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #1B5E20;
}
.stButton>button {
    background-color: #2E7D32;
    color: white;
    border-radius: 8px;
    height: 3em;
}
.stButton>button:hover {
    background-color: #1B5E20;
}
.sidebar .sidebar-content {
    background-color: #E8F5E9;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def get_model():
    return load_model(device="cpu")

model, classes = get_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("♻️ ScrapNet")
st.sidebar.markdown("### Model Information")

st.sidebar.markdown("""
**Architecture:** EfficientNet-B0  
**Framework:** PyTorch  
**Image Size:** 224x224  
**Classes:** {}  
**Deployment:** Streamlit Cloud  
""".format(len(classes)))

st.sidebar.markdown("---")
st.sidebar.markdown("### Training Details")
st.sidebar.markdown("""
- Transfer Learning  
- Pretrained on ImageNet  
- Fine-tuned classifier head  
- Cross-Entropy Loss  
""")

# ---------------- MAIN HEADER ----------------
st.title("♻️ ScrapNet – Waste Classification System")
st.markdown("A deep learning system for intelligent waste categorization using transfer learning.")

# ---------------- MODEL DETAILS ----------------
st.header("🔬 Model Architecture")

st.markdown("""
ScrapNet uses **EfficientNet-B0**, a state-of-the-art convolutional neural network
known for its efficiency and performance.

The classifier head was modified to match the number of waste categories,
and the network was fine-tuned on a labeled waste dataset.
""")

# ---------------- PERFORMANCE SECTION ----------------
st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Accuracy Curve**")
    show_artifact_image("accuracy_curve.png")

with col2:
    st.markdown("**Loss Curve**")
    show_artifact_image("loss_curve.png")

st.markdown("**Confusion Matrix (Normalized)**")
show_artifact_image("confusion_matrix_normalized.png")

# ---------------- PREDICTION SECTION ----------------
st.header("🧪 Run Inference")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    label, conf, probs = predict_image(model, classes, img, device="cpu")

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{conf:.2%}**")

    topk = np.argsort(probs)[::-1][:3]
    st.markdown("**Top 3 Predictions:**")
    for i in topk:
        st.write(f"- {classes[i]}: {probs[i]:.2%}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("© 2026 ScrapNet | AI for Sustainable Waste Management 🌱")
