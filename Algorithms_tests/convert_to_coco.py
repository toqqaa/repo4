


import os
import json

def convert_yolo_to_coco(yolo_annotations, annotation_id_counter):
    coco_annotations = {"images": [], "annotations": [], "categories": []}

    for image_filename, boxes in yolo_annotations.items():
        # Generate a unique image_id using hash function or another method
        image_id = hash(image_filename)
        
        # Add image information to the 'images' key
        image_info = {
            "id": image_id,
            "file_name": f"{image_filename}.jpg",
        }  # Assuming image filenames are "{image_filename}.jpg"
        coco_annotations["images"].append(image_info)

        image_annotations = []
        for box in boxes:
            # Ensure the YOLO box has enough elements
            if len(box) >= 5:
                x, y, w, h, score = box[1], box[2], box[3], box[4], box[5]
                annotation = {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": int(box[0]),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": score,
                }
                image_annotations.append(annotation)
                annotation_id_counter += 1

                # Add category if not already in the list
                category_id = int(box[0])
                if category_id not in [cat["id"] for cat in coco_annotations["categories"]]:
                    coco_annotations["categories"].append(
                        {
                            "id": category_id,
                            "name": f"class_{category_id}",
                            "supercategory": "unknown",
                        }
                    )
            else:
                print(
                    f"Warning: Skipping box without enough elements in image {image_id}"
                )

        coco_annotations["annotations"].extend(image_annotations)

    return coco_annotations, annotation_id_counter

# ... (rest of your code remains unchanged)


# ... (rest of your code remains unchanged)



def load_annotations_from_text_file(file_path):
    annotations = {}
    with open(file_path, "r") as f:
        for line in f:
            values = line.strip().split()
            image_id = os.path.splitext(os.path.basename(file_path))[0]
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(list(map(float, values)))
    return annotations


def convert_yolo_files_to_coco(directory_path):
    coco_annotations = {"images": [], "annotations": [], "categories": []}
    annotation_id_counter = 1

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            yolo_annotations = load_annotations_from_text_file(file_path)
            coco_format, annotation_id_counter = convert_yolo_to_coco(
                yolo_annotations, annotation_id_counter
            )
            coco_annotations["images"].extend(coco_format["images"])
            coco_annotations["annotations"].extend(coco_format["annotations"])
            coco_annotations["categories"].extend(coco_format["categories"])

    return coco_annotations


def save_coco_annotations_to_json(coco_annotations, save_path):
    with open(save_path, "w") as json_file:
        json.dump(coco_annotations, json_file, indent=4)


# Example usage
yolo_directory = r"/home/toqa/projects/Algorithms_test/yolo_output_T"
coco_annotations = convert_yolo_files_to_coco(yolo_directory)
save_coco_annotations_to_json(
    coco_annotations,
    "/home/toqa/projects/Algorithms_test/coco_annotations_Predicted.json",
)