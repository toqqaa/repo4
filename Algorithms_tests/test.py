import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_annotations(coco, image_id):
    # Load image information
    img_info = coco.loadImgs(image_id)[0]
    img_path = "/home/toqa/projects/Algorithms_test/OIDv4_ToolKit/OID/Dataset/validation/Person" + img_info['0ca4cd34574b30be.txt']
    image = Image.open(img_path)

    # Load annotations for the specified image
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)

    # Visualize the image
    plt.imshow(image)

    # Plot bounding boxes
    ax = plt.gca()
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1] - 5, category_name, color='r')

    plt.show()

# Load your COCO format annotations
coco_annotations_path = "/home/toqa/projects/Algorithms_test/coco_annotations_ground_truth.json"
coco = COCO(coco_annotations_path)

# Choose an image ID to visualize
image_id_to_visualize = 1  # Change this to the desired image ID
visualize_annotations(coco, image_id_to_visualize)



# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# def check_bounding_box_coordinates(annotation, image_width):
#     bbox = annotation['bbox']
#     return all(0 <= coord <= image_width for coord in bbox)

# # Load ground truth and predicted annotations using COCO API
# coco_gt = COCO(
#     r"/home/toqa/projects/Algorithms_test/coco_annotations_ground_truth.json"
# )

# coco_pred = COCO(
#     r"/home/toqa/projects/Algorithms_test/coco_annotations_Predicted.json"
# )

# image_width = 1920  # Replace with your actual image width

# # Filter out annotations with invalid bounding box coordinates
# valid_gt_annotations = [gt_annotation for gt_annotation in coco_gt.dataset['annotations']
#                          if check_bounding_box_coordinates(gt_annotation, image_width)]

# valid_pred_annotations = [pred_annotation for pred_annotation in coco_pred.dataset['annotations']
#                           if check_bounding_box_coordinates(pred_annotation, image_width)]

# # Ensure that category IDs match
# gt_category_ids = set([ann['category_id'] for ann in valid_gt_annotations])
# pred_category_ids = set([ann['category_id'] for ann in valid_pred_annotations])
# print("Ground Truth Category IDs:", gt_category_ids)
# print("Predicted Category IDs:", pred_category_ids)

# # Convert predicted bounding box coordinates to pixel values
# for pred_annotation in valid_pred_annotations:
#     pred_annotation['bbox'][0] *= image_width
#     pred_annotation['bbox'][1] *= image_width
#     pred_annotation['bbox'][2] *= image_width
#     pred_annotation['bbox'][3] *= image_width

# # Perform evaluation using the default COCO evaluation script
# coco_eval_default = COCOeval(coco_gt, coco_pred, 'bbox')
# coco_eval_default.evaluate()
# coco_eval_default.accumulate()
# coco_eval_default.summarize()
