
    
import cv2
import time
from ultralytics import YOLO  # Make sure you have the ultralytics library installed
import torch

# Load the YOLO model
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Open a video file or use a camera (0 for default camera)
video_path = r"/home/toqa/projects/Algorithms_test/Algorithms_tests/people.mp4"
cap = cv2.VideoCapture(video_path)

# Define class mapping (replace this with your actual class mapping)
class_mapping = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# Variables for calculating FPS
frame_count = 0
start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video is over

    # Run YOLO inference on the frame
    results = model(frame)

    for result in results:
        for i, box in enumerate(result):
            box = result.boxes.xyxy[i]
            x, y, width, height = (
                int(box[0].item()),
                int(box[1].item()),
                int(box[2].item()),
                int(box[3].item()),
            )
            confidence = result.boxes.conf[i].item()
            class_prediction = result.boxes.cls[i].item()  # Indexing based on 'i'

            # Replace class name with class id
            class_name = class_mapping.get(
                class_prediction, f"UnknownClass_{class_prediction}"
            )

            cv2.rectangle(frame, (x, y), (width, height), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{class_name}, {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= 1.0:  # Update FPS every second
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = current_time

    # Display the frame with FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (int(frame.shape[1] - 150), 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Display the frame
    cv2.imshow("YOLO Inference", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
