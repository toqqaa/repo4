import cv2
import torch
from torchvision import transforms 
from torchvision.transforms import functional as F
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# Load YOLOv8 model (you need to adjust the path)
model_path = "/home/toqa/projects/Algorithms_test/yolov8n.pt"
model = YOLO("yolov8n.pt")

# Load video
video_path = "/home/toqa/projects/Algorithms_test/Algorithms_tests/people.mp4"
cap = cv2.VideoCapture(video_path)
coco_annotations = {"images": [], "annotations": [], "categories": []}
annotation_id_counter = 1

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the input frame
    input_tensor = F.to_tensor(F.resize(F.to_pil_image(frame), (640, 640)))
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform object detection
    with torch.no_grad():
        detections = model(input_tensor)

    # Post-process detections
    # Ensure that the structure matches the output of your YOLOv8 model
    # Commonly, you have 'boxes', 'scores', and 'labels' in the output
    boxes = detections.xyxy[0][:, :4].cpu().numpy()
    scores = detections.xyxy[0][:, 4].cpu().numpy()
    labels = detections.xyxy[0][:, 5].cpu().numpy()

    # Save detection results to COCO format
    image_info = {"id": annotation_id_counter, "file_name": f"{annotation_id_counter}.jpg"}
    coco_annotations["images"].append(image_info)

    for box, score, label in zip(boxes, scores, labels):
        x, y, w, h = box
        annotation = {
            "id": annotation_id_counter,
            "image_id": annotation_id_counter,
            "category_id": int(label),
            "bbox": [float(x), float(y), float(w - x), float(h - y)],
            "area": float((w - x) * (h - y)),
            "iscrowd": 0,
        }
        coco_annotations["annotations"].append(annotation)
        annotation_id_counter += 1


# Save COCO annotations to a file
with open("coco_annotations.json", "w") as json_file:
    json.dump(coco_annotations, json_file, indent=4)

# Load ground truth annotations
gt_coco = COCO("coco_annotations.json")

# Load detection results (you need to adjust the path)
results_coco = gt_coco.loadRes("detection_results.json")

# Create COCOeval object
coco_eval = COCOeval(gt_coco, results_coco, "bbox")

# Run evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Access mAP
mAP = coco_eval.stats[1]
print(f"Mean Average Precision (mAP): {mAP}")
