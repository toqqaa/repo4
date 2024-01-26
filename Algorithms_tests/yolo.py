from ultralytics import YOLO
import cv2
import os

# Path to the directory containing images
images_path = r"/home/toqa/projects/Algorithms_test/OIDv4_ToolKit/OID/Dataset/validation/Person"

# Output directory for text files
output_directory = "yolo_output_T"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

model = YOLO("yolov8n.pt")

# Process results for each image in the directory
for filename in os.listdir(images_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_path, filename)

        # Run YOLO inference
        results = model(image_path, save_json=True)

        # Output text file path for the current image
        output_file_path = os.path.join(
            output_directory, f"{os.path.splitext(filename)[0]}.txt"
        )

        # Write detected objects information to the text file
        with open(output_file_path, "w") as f:
            for result in results:
                for i, box in enumerate(result):
                    box = result.boxes.xywhn[i]
                    x, y, width, height = (
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                    )
                    confidence = result.boxes.conf[i].item()

                    line = f"{int(result.boxes.cls[i])} {x} {y} {width} {height} {confidence:.4f}\n"
                    f.write(line)
                    print(line)

# Display the images
cv2.waitKey(0)
cv2.destroyAllWindows()