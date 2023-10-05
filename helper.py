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


def display_tracker_options():
    #display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = 'bytetrack.yaml'
        return is_display_tracker, tracker_type
    return is_display_tracker, tracker_type


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
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
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_webcam(conf, model):
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
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    #if st.sidebar.button('Detect Objects'):
    try:
        vid_cap = cv2.VideoCapture(source_webcam)
        #st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf,
                                         model,
                                         st_frame,
                                         image,
                                         is_display_tracker,
                                         tracker,
                                         )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.sidebar.error("Error loading video: " + str(e))

def webcam_old(conf, model):
    
    is_display_tracker, tracker = display_tracker_options()
    
    webrtc_ctx = webrtc_streamer(
        key="object_detection", 
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
            # Read frame from IP camera
            ret, frame = cap.read()

            # Display the frame in Streamlit
            if ret:
                webrtc_ctx.video_receiver.process_frame(frame)
                st.image(frame, channels="BGR")
                _display_detected_frames(conf,
                                         model,
                                         st_frame,
                                         frame,
                                         is_display_detected_frames,
                                         tracker,
                                         )
            
        # Release the VideoCapture object and cleanup
        cap.release()
        cv2.destroyAllWindows()
        
def webcam(conf, model):
    
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
            detected_frame = model.track(frame)

            # Display the result in Streamlit
            webrtc_ctx.video_receiver.process_frame(detected_frame)
            st.image(detected_frame, channels="BGR", use_column_width=True)            
            
            #if ret:
            #    webrtc_ctx.video_receiver.process_frame(frame)
            #    st.image(frame, channels="BGR")
            
        # Release the VideoCapture object and cleanup
        cap.release()
        cv2.destroyAllWindows()