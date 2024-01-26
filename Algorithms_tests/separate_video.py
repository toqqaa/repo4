import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# Load your custom-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load video
video_path = r"/home/toqa/projects/Algorithms_test/Algorithms_tests/people.mp4"
cap = cv2.VideoCapture(video_path)

# Set output directory
output_dir = "output_images"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Process video frames
count = 0
while cap.isOpened():
    # Read frame from video
    ret, frame = cap.read()
    if ret:
        # Run frame through YOLOv8 model
        results = model.predict(
            frame, save=True, conf=0.5
        )  # You may adjust parameters as needed

        if results[0] is not None:
            # Plot annotated frame with bounding boxes
            annotated_frame = results[0].plot()

            # Save annotated frame as an image
            name = f"{output_dir}/frame_{count}.jpg"
            image = np.array(annotated_frame)
            cv2.imwrite(name, image)
            count += 1
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
