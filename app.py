import streamlit as st
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
from PIL import Image
import cv2
import tensorflow as tf
import tempfile, os
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "best_model.keras"
CLASS_NAMES = ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(arr):
    img = cv2.resize(arr.astype(np.uint8), (224, 224))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return np.expand_dims(img / 255.0, axis=0)

st.set_page_config(page_title="Pneumonia Detector", page_icon="x-ray")
st.title("Pneumonia Detection")
st.write("Upload a chest X-ray (.dcm, .png, or .jpg)")

uploaded = st.file_uploader("Choose file", type=["dcm","png","jpg","jpeg"])
if uploaded:
    if uploaded.name.endswith(".dcm"):
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        ds = pydicom.dcmread(tmp_path, force=True)
        if not hasattr(ds, 'file_meta') or ds.file_meta is None:
            ds.file_meta = pydicom.Dataset()
        if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        arr = ds.pixel_array.astype(np.float32)
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    else:
        arr = np.array(Image.open(uploaded).convert("RGB"))
    st.image(arr, caption="Uploaded X-ray", use_container_width=True, clamp=True)
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Upload best_model.keras to /workspaces/Pneumonia-Detection/")
    else:
        with st.spinner("Running inference..."):
            model = load_model()
            probs = model.predict(preprocess(arr))[0]
            pred = CLASS_NAMES[int(np.argmax(probs))]
        st.success(f"Prediction: {pred}")
        for cls, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f"{cls}: {p:.1%}")
