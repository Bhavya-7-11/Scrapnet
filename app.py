import streamlit as st
from PIL import Image
import numpy as np

from predict import load_model, predict_image

st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")

@st.cache_resource
def get_model():
    return load_model(device="cpu")

model, classes = get_model()

st.title("♻️ Waste Classification Demo")
st.write("Upload an image and the model will predict the waste category.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    label, conf, probs = predict_image(model, classes, img, device="cpu")

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{conf:.2%}**")

    topk = np.argsort(probs)[::-1][:3]
    st.write("Top 3 predictions:")
    for i in topk:
        st.write(f"- {classes[i]}: {probs[i]:.2%}")