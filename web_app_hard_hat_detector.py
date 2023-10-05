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
<video id="webcam" width="640" height="480" autoplay></video>
<canvas id="output" width="640" height="480"></canvas>
<script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('output');
    const ctx = canvas.getContext('2d');

    // Initialize webcam
    async function setupWebcam() {
        const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });
        video.srcObject = stream;
    }
    setupWebcam();

    // Load YOLO model (replace 'modelPath' with your model's URL)
    const modelPath = model_path;
    let model;

    async function loadModel() {
        model = await tf.loadGraphModel(modelPath + 'model.json');
    }
    loadModel();

    // Function to run object detection
    async function runObjectDetection() {
        while (true) {
            await tf.nextFrame();
            const img = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(img, [416, 416]);
            const casted = resized.cast('float32');
            const expanded = casted.expandDims(0);
            const predictions = await model.predict(expanded);

            // Process predictions and draw bounding boxes
            // You'll need to implement this part based on your model
            // For simplicity, consider using tfjs-yolo-tiny or similar libraries for YOLO detection
        }
    }
    runObjectDetection();
</script>
"""

# Display the webcam and object detection canvas using HTML
st.components.v1.html(html_code)