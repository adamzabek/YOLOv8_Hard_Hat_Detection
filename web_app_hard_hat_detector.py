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
custom_css = """
<style>
h1 {
    text-align: center;
    margin-top: 50px; /* Adjust the margin to vertically center the title */
}
</style>
"""

# Use the custom CSS
st.write(custom_css, unsafe_allow_html=True)

st.title("Hard Hat Detector")

custom_css = """
<style>
.centered-text {
    text-align: center;
}
</style>
"""

st.write(custom_css, unsafe_allow_html=True)

st.markdown("<div class='centered-text'>Select Model Confidence</div>", unsafe_allow_html=True)

confidence = float(st.slider("", 25, 100, 40)) / 100
    
# Load Pre-trained ML Model
model_path = Path(settings.DETECTION_MODEL)

try:
    model = YOLO(model_path)
    
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}.")
    st.error(ex)
    
helper.play_webcam(confidence, model)