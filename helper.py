from ultralytics import YOLO
import time
import streamlit as st
import cv2

import settings
from streamlit_webrtc import WebRtcMode, webrtc_streamer


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def play_webcam(conf, model):
    
    #tracker = cv2.TrackerKCF_create()
    
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
        cap = cv2.VideoCapture(settings.WEBCAM_PATH)

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Check if the frame was read successfully
            if not ret:
                print("Error: Could not read frame.")
                break
        
            results = model.track(source = frame, show = True, persist = True, tracker="bytetrack.yaml")

            webrtc_ctx.video_receiver.process_frame(results)
            
            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the VideoCapture and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()