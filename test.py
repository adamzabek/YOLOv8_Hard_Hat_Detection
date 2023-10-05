from ultralytics import YOLO
import cv2


model = YOLO('./model/best.pt')


# Create a VideoCapture object to access the camera (usually the default camera is 0)
cap = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture and display frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break
        
    results = model.track(source = frame, show = True, persist = True, tracker="bytetrack.yaml")
    #res_plotted = results[0].plot()
    # Display the frame in a window
    #cv2.imshow('Camera Feed', res_plotted)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()