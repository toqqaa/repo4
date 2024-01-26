import numpy as np
import cv2

def calculate_iou(boxa, boxb):
    box1 = (50, 50, 150, 150)  # (x1, y1, x2, y2) coordinates of the first bounding box

    box2 = (100, 100, 200, 200)  # (x1, y1, x2, y2) coordinates of the second bounding box

    iou_value = calculate_iou(box1, box2)
    print(f"IOU value: {iou_value:.2f}")

    x1_min, y1_min, x1_max, y1_max = boxa
    x2_min, y2_min, x2_max, y2_max = boxb
    
    # Calculate the coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    # Calculate the area of the intersection
    inter_width = max(0, x_inter_max - x_inter_min + 1)
    inter_height = max(0, y_inter_max - y_inter_min + 1)
    intersection_area = inter_width * inter_height
    
    # Calculate the areas of the bounding boxes
    boxa_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    boxb_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
    
    # Calculate the area of union
    union_area = boxa_area + boxb_area - intersection_area
    
    # Calculate and return IOU
    iou = intersection_area / union_area
    return iou