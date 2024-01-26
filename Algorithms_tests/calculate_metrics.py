
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt = COCO(
    r"/home/toqa/projects/Algorithms_test/coco_annotations_ground_T.json"
)

coco_pred = COCO(""
    r"/home/toqa/projects/Algorithms_test/coco_annotations_predected_T.json"
)


coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

mAP_50_to_95 = coco_eval.stats[0]
mAP_50 = coco_eval.stats[1]
mAR_50_to_95 = coco_eval.stats[8]
print("mAP_50_to_95 :", mAP_50_to_95)
print("mAP_50: ", mAP_50)
print("mAR_50_to_95: ", mAR_50_to_95)
