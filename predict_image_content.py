# About:    Simple code to use the YOLO 8 for finding the contents of a given image.
# Notice:   Some parts of this code were generated using AI tools.

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose other model weights like yolov8s.pt, yolov8m.pt, etc.

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    for result in results:
        # Each result contains a box with xyxy format, confidence, and class id
        x1, y1, x2, y2 = result[:4]
        conf = result[4]
        cls = result[5]
        label = f"{model.names[int(cls)]} {conf:.2f}"
        # Draw rectangle
        if hasattr(cv2, 'rectangle'):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        else:
            raise AttributeError("cv2 does not have the attribute 'rectangle'")
        # Put label
        if hasattr(cv2, 'putText'):
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            raise AttributeError("cv2 does not have the attribute 'putText'")
    return image

# Load an image
# image_path = 'img/anim_01.jpeg'
image_path = 'img/cat_01.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image from path: {image_path}")

# Detect objects in the image
results = model.predict(image)
