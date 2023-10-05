import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import settings
import helper
import av
import logging
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from aiortc.contrib.media import MediaRecorder

import cv2
from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)

# Setting page layout
st.set_page_config(
    page_title="Hard Hat Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Hard Hat Detector")

# Sidebar
st.header("ML Model Config")

confidence = float(st.slider(
    "Select Model Confidence", 25, 100, 40)) / 100    
    
# Load Pre-trained ML Model
model_path = Path(settings.DETECTION_MODEL)
#model_path = './model/best.pt'

try:
    model = YOLO(model_path)
    
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}.")
    st.error(ex)
    
try:
    helper.webcam(confidence, model)
except Exception as ex:
    st.error(f"Unable to run webcamera.")
    st.error(ex)