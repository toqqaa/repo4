from ultralytics import YOLO
import cv2
import torch

path = r"/home/toqa/projects/Algorithms_test/Algorithms_tests/people.mp4"
model = YOLO("yolov8n.pt")

# Run YOLO inference
results = model(path, show=True, line_thickness=2)

# Optionally, you can check if GPU is available
if torch.cuda.is_available():
    print("Using GPU for inference.")
else:
    print("GPU not available. Using CPU.")

cv2.waitKey(0)
cv2.destroyAllWindows()
