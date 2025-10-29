# streamlit_app.py
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
from PIL import Image
import time
import base64

st.set_page_config(page_title="DeepVision Analyzer", layout="wide", page_icon="🤖")

def add_bg_video(video_path):
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
            color: white;
        }}
        video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            filter: brightness(30%);
        }}
        </style>
        <video autoplay muted loop>{base64.b64encode(video_bytes).decode()}</video>
        """,
        unsafe_allow_html=True,
    )



# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
    <style>
    .big-title {
        text-align: center;
        font-size: 50px;
        font-weight: 700;
        color: #4A90E2;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #B0BEC5;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4A90E2, #9013FE);
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 30px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(45deg, #9013FE, #4A90E2);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# DETECTION APP
# -----------------------------------
def detection_app():
    st.markdown("<h1 class='big-title'>DeepVision Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Perform Real-Time Detection | Segmentation | Pose Estimation</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")

        model_option = st.selectbox("Select Model Type", ["Detection", "Segmentation", "Pose"])
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

        if model_option == "Detection":
            model_path = "yolov8n.pt"
        elif model_option == "Segmentation":
            model_path = "yolov8n-seg.pt"
        else:
            model_path = "yolov8n-pose.pt"

        st.info(f"Model Loaded: `{model_path}`")
        model = YOLO(model_path)

        input_source = st.radio("Select Input Type", ["📷 Webcam", "🎞️ Video Upload", "🖼️ Image Upload"])

    # IMAGE MODE
    if input_source == "🖼️ Image Upload":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Run Detection"):
                with st.spinner("Processing Image..."):
                    results = model.predict(np.array(image), conf=confidence)
                    annotated = results[0].plot()
                    st.image(annotated, caption="Detection Output", use_container_width=True)
                    st.download_button("Download Result", data=cv2.imencode('.jpg', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))[1].tobytes(), file_name="output.jpg")

    # VIDEO MODE
    elif input_source == "🎞️ Video Upload":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            if st.button("Start Detection"):
                cap = cv2.VideoCapture(video_path)
                stframe = st.empty()
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed = 0
                progress = st.progress(0)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output_path = "processed_output.mp4"
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, conf=confidence)
                    annotated = results[0].plot()
                    writer.write(annotated)
                    processed += 1
                    progress.progress(processed / frame_count)
                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                cap.release()
                writer.release()
                st.success("✅ Video Detection Complete!")
                with open(output_path, "rb") as f:
                    st.download_button("Download Processed Video", f, file_name="result_yolo.mp4")

    # WEBCAM MODE
    elif input_source == "📷 Webcam":
        st.write("Click below to start your webcam feed.")
        if st.button("Start Webcam Detection"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=confidence)
                annotated = results[0].plot()
                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            cap.release()

    st.write("---")
    st.markdown("<p style='text-align:center; | DeepVision Analyzer | Powered by YOLOv8</p>", unsafe_allow_html=True)

detection_app()
