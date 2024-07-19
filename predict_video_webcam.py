# About:    Simple code to use the YOLO 8 for finding the content of the live video generated from your webcam.
# Notice:   Some parts of this code were generated using AI tools.

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load the YOLOv8 model

cap = cv2.VideoCapture(0) # Initialize webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()


    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Convert results to an annotated image
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 based Webcam Predictions', annotated_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

