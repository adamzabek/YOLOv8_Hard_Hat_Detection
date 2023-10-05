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


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    #image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    res = model.track(image, conf=conf, show=True, tracker="bytetrack.yaml")

    # Plot the detected objects on the video frame
    #res_plotted = res[0].plot()
    webrtc_ctx.video_receiver.process_frame(res)
    st_frame.image(res,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
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
        try:
            vid_cap = cv2.VideoCapture(settings.WEBCAM_PATH)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        
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

        # Read frame from IP camera
        ret, frame = cap.read()            
            
        if ret:
            detected_frame = model.track(source = frame, persist=True, tracker="bytetrack.yaml")

            # Display the result in Streamlit
            webrtc_ctx.video_receiver.process_frame(detected_frame)
            st.image(detected_frame, channels="BGR")           
            
            #if ret:
            #    webrtc_ctx.video_receiver.process_frame(frame)
            #    st.image(frame, channels="BGR")
            
        # Release the VideoCapture object and cleanup
        cap.release()
        cv2.destroyAllWindows()