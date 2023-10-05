import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import settings
import av
from turn import get_ice_servers
import logging
import queue
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from aiortc.contrib.media import MediaRecorder
import cv2
from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)


webrtc_ctx = webrtc_streamer(
    key="example", 
    video_processor_factory=None,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

if webrtc_ctx.video_receiver:
    # Initialize the VideoCapture object for IP camera
    cap = cv2.VideoCapture("0")

    while True:
        # Read frame from IP camera
        ret, frame = cap.read()

        # Display the frame in Streamlit
        if ret:
            webrtc_ctx.video_receiver.process_frame(frame)
            st.image(frame, channels="BGR")

    # Release the VideoCapture object and cleanup
    cap.release()
    cv2.destroyAllWindows()