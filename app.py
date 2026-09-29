from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

WEIGHTS = Path(__file__).parent / "best.pt"
TRAIN_IMGSZ = 1536

st.set_page_config(page_title="ParasiteNet", page_icon="🔬", layout="wide")
st.title("ParasiteNet: Blood Parasite Detector")
st.caption("YOLO26m-seg instance segmentation of RBCs, Babesia, Plasmodium, and Toxoplasma "
           "in Giemsa-stained blood smears.")


@st.cache_resource
def load_model():
    return YOLO(str(WEIGHTS))


if not WEIGHTS.is_file():
    st.error("Model weights (best.pt) were not found next to app.py.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    imgsz = st.select_slider("Inference image size", options=sorted({640, 1024, TRAIN_IMGSZ}),
                             value=TRAIN_IMGSZ)
    st.caption("Smaller sizes run faster on the free server but are less accurate.")

uploaded = st.file_uploader("Upload a blood smear image",
                            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("That file could not be read as an image.")
    st.stop()

with st.spinner("Running the model..."):
    try:
        r = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

left, right = st.columns(2)
left.image(image, caption="Original")
right.image(r.plot()[:, :, ::-1], caption="Detections (masks + boxes)")

names = [r.names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else []
if not names:
    st.warning("Nothing detected at this confidence threshold. Try lowering it.")
    st.stop()

st.subheader("Count per class")
counts = Counter(names)
for col, (cls, n) in zip(st.columns(len(counts)), sorted(counts.items())):
    col.metric(cls, n)

st.subheader("Every detection")
st.dataframe(pd.DataFrame({"Class": names,
                           "Confidence": [round(float(s), 3) for s in r.boxes.conf.tolist()]}),
             hide_index=True)
st.caption("Research prototype for a class project. Not a medical diagnostic tool.")
