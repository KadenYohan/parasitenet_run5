from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

WEIGHTS = Path(__file__).parent / "ParasiteNet_Aentropyda.pt"
TRAIN_IMGSZ = 1024
MAX_IMAGES = 10   # the free server runs on CPU, so keep batches small

st.set_page_config(page_title="ParasiteNet_Aentropyda", page_icon="🔬", layout="wide")
st.title("ParasiteNet_Aentropyda: Blood Parasite Detector")
st.caption("Instance segmentation of red blood cells, Babesia, Plasmodium, and Toxoplasma "
           "in Giemsa-stained blood smears.")


@st.cache_resource
def load_model():
    return YOLO(str(WEIGHTS))


if not WEIGHTS.is_file():
    st.error(f"Model weights ({WEIGHTS.name}) were not found next to app.py.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

CLASS_NAMES = list(model.names.values())

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    imgsz = st.select_slider("Inference image size", options=sorted({640, 1024, TRAIN_IMGSZ}),
                             value=TRAIN_IMGSZ)
    smooth = st.checkbox("Full-resolution masks (smoother edges)", value=True)
    st.caption("Smaller sizes and turning off full-resolution masks run faster "
               "on the free server but are less precise.")

uploads = st.file_uploader("Upload blood smear images (you can select several at once)",
                           type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
                           accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more images to start.")
    st.stop()

if len(uploads) > MAX_IMAGES:
    st.warning(f"{len(uploads)} images uploaded. Only the first {MAX_IMAGES} will be processed.")
    uploads = uploads[:MAX_IMAGES]

summary_box = st.container()          # filled in after all images are processed
summary_rows = []
progress = st.progress(0.0, text="Starting...")
tabs = st.tabs([f"{i + 1}. {up.name}" for i, up in enumerate(uploads)])

for i, (tab, up) in enumerate(zip(tabs, uploads)):
    progress.progress(i / len(uploads), text=f"Processing {up.name} ({i + 1}/{len(uploads)})")
    with tab:
        try:
            image = Image.open(up).convert("RGB")
        except Exception:
            st.error("This file could not be read as an image.")
            continue

        try:
            r = model.predict(image, imgsz=imgsz, conf=conf, retina_masks=smooth, verbose=False)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            continue

        left, right = st.columns(2)
        left.image(image, caption="Original")
        right.image(r.plot()[:, :, ::-1], caption="Detections (masks + boxes)")

        names = [r.names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else []
        counts = Counter(names)
        summary_rows.append({"Image": up.name, **{c: counts.get(c, 0) for c in CLASS_NAMES},
                             "Total": len(names)})

        if not names:
            st.warning("Nothing detected at this confidence threshold. Try lowering it.")
            continue

        st.subheader("Count per class")
        for col, (cls, n) in zip(st.columns(len(counts)), sorted(counts.items())):
            col.metric(cls, n)

        st.subheader("Every detection")
        st.dataframe(pd.DataFrame({"Class": names,
                                   "Confidence": [round(float(s), 3) for s in r.boxes.conf.tolist()]}),
                     hide_index=True)

progress.empty()

with summary_box:
    if len(summary_rows) > 1:
        st.subheader(f"Summary of all {len(summary_rows)} images")
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
        st.caption("Click each tab below to see that image's detections.")

st.caption("Research prototype for a class project. Not a medical diagnostic tool.")
