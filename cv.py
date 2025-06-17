import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

# ✅ Set Streamlit page config FIRST
st.set_page_config(page_title="🔥 Fire & Smoke Detection", layout="wide")

# Load YOLOv8 model once
@st.cache_resource
def load_model():
    return YOLO("trainmodel.pt")

model = load_model()

# Title and subtitle
st.markdown("""
    <h1 style='text-align: center; color: #dc3545;'>🔥 Real-Time Fire and Smoke Detection</h1>
    <p style='text-align: center; color: #6c757d;'>Upload an image or use your webcam to detect <strong>fire</strong> and <strong>smoke</strong> using YOLOv8</p>
""", unsafe_allow_html=True)

# Tabs for upload or webcam
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Webcam"])

# ---- TAB 1: Image Upload ----
with tab1:
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("🚀 Running detection..."):
            results = model.predict(source=tmp_path, save=False, conf=0.4)

            # Draw bounding boxes
            image = cv2.imread(tmp_path)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="✅ Detection Result", use_container_width=True)

# ---- TAB 2: Real-time Webcam Detection ----
with tab2:
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([], use_container_width=True)
    cam = cv2.VideoCapture(0)

    if run:
        st.info("Press 'Stop Webcam' to release the camera.")

        while run:
            success, frame = cam.read()
            if not success:
                st.error("Failed to access webcam.")
                break

            # Resize for speed
            resized = cv2.resize(frame, (640, 480))
            results = model.predict(source=resized, save=False, conf=0.4, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(resized, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame)

        cam.release()
    else:
        st.warning("👆 Check the box to start webcam stream.")
