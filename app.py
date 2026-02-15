import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from predict import load_model, predict_image

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="ScrapNet • Waste Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------- Styling (CSS) ----------------
st.markdown(
    """
    <style>
      .main { padding-top: 1.2rem; }
      .hero {
        text-align:center;
        padding: 1.2rem 1.2rem 0.4rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(25,135,84,0.15), rgba(13,110,253,0.10));
        border: 1px solid rgba(255,255,255,0.08);
      }
      .hero h1 { margin-bottom: 0.2rem; }
      .subtle { opacity: 0.85; }
      .card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
      }
      .small { font-size: 0.92rem; opacity: 0.85; }
      .badge {
        display:inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: rgba(13,110,253,0.18);
        border: 1px solid rgba(13,110,253,0.35);
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
st.sidebar.markdown("## ♻️ ScrapNet")
st.sidebar.markdown(
    """
**Waste Classification using EfficientNet-B0**

**How to use**
1. Upload a JPG/PNG/WEBP  
2. Wait ~1–2 seconds  
3. Get class + confidence + Top-3

**Tips**
- Try clear objects (plastic bottle, paper, etc.)
- If you’re on mobile, use a smaller image (<1MB)
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
topk_n = st.sidebar.slider("Show Top-K predictions", 3, 5, 3)
show_probs = st.sidebar.toggle("Show probability table", value=False)

# ---------------- Load model once ----------------
@st.cache_resource
def get_model():
    return load_model(device="cpu")

model, classes = get_model()

# ---------------- Hero ----------------
st.markdown(
    """
    <div class="hero">
      <h1>♻️ ScrapNet</h1>
      <div class="subtle">Upload an image and the model predicts the waste category.</div>
      <div style="margin-top:0.6rem;">
        <span class="badge">EfficientNet-B0 • Transfer Learning</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------- Upload Card ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload an image")
uploaded = st.file_uploader(
    "Supported: JPG, JPEG, PNG, WEBP",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('<div class="small">Tip: Use a clear object image for best results.</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------- Prediction ----------------
if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Preview")
        st.image(img, caption="Uploaded image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        label, conf, probs = predict_image(model, classes, img, device="cpu")

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✅ Prediction")

        c1, c2 = st.columns(2)
        c1.metric("Predicted Class", label)
        c2.metric("Confidence", f"{conf*100:.2f}%")

        # Top-K
        probs = np.array(probs)
        k = min(topk_n, len(classes))
        topk_idx = np.argsort(probs)[::-1][:k]

        df = pd.DataFrame({
            "Class": [classes[i] for i in topk_idx],
            "Probability": [float(probs[i]) for i in topk_idx],
        })

        st.write("")
        st.markdown("#### 📊 Top predictions")
        st.bar_chart(df.set_index("Class")["Probability"])

        if show_probs:
            st.markdown("#### 🧾 Probability table")
            st.dataframe(df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception:
        st.error("Couldn’t process that image. Try a different file (smaller/clearer).")

else:
    st.info("Upload an image to get a prediction.")
