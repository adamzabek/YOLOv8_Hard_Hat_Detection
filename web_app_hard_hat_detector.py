import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import settings

# Load YOLO model
model_path = Path(settings.DETECTION_MODEL)

# Set the title of the Streamlit app
st.title("Real-Time YOLO Object Detection in Streamlit")

# HTML code to access webcam and display video
html_code = """
<video id="webcam" width="100%" height="100%" autoplay></video>
<style>
    body {
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    #webcam {
        object-fit: cover;
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0;
        left: 0;
    }
</style>
<script>
    const video = document.getElementById('webcam');

    // Initialize webcam
    async function setupWebcam() {
        const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });
        video.srcObject = stream;
    }
    setupWebcam();

</script>
"""

# Display the webcam and object detection canvas using HTML
st.components.v1.html(html_code)