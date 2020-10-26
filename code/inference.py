from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import sys
import json
import random
import cv2
import glob
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Pansharpen images")
parser.add_argument("input", help="Input dataset file")
parser.add_argument("output", help="Output predictions file")
parser.add_argument('--viz', default="", help="Vizualize prediction images")
args = parser.parse_args(sys.argv[1:])



def bbox_to_circle(bbox, sampling=100):
    x1, y1, x2, y2 = bbox
    r = (x2 - x1) / 2
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    t = np.linspace(0, 2*np.pi, sampling)
    x = r * np.cos(t) + mx
    y = r * np.sin(t) + my
    return np.vstack((x, y)).T


dataset = args.input
output = args.output
viz = args.viz
    
# dataset = "/media/DATA1/Topcoder/circle_finder/circle/labels_valid.json"
# dataset = "/media/DATA1/Topcoder/circle_finder/pansharpen/labels_valid.json"

# dataset = "/home/mzins/dev/Circle_Finder/detectron_labels_valid.json"
# dataset = "/home/mzins/dev/Circle_Finder/detectron_labels_test.json"

# file_out = "prediction_test.json"
# file_out = "prediction_valid.json"

# output = "/home/mzins/dev/detectron2/predictions"
# output = ""



with open(dataset, "r") as fin:
    data = json.load(fin)




cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_cpu.pth")
cfg.MODEL.WEIGHTS = "checkpoint/model_final.pth"
SCORE_THRESHOLD = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
# cfg.MODEL.PIXEL_MEAN = [102.28063327, 93.66882446, 92.7146011]
cfg.MODEL.PIXEL_MEAN = [111.01797572, 102.54100801, 93.86145873]
#cfg.MODEL.PIXEL_MEAN = [116.05156287, 116.05156287, 116.05156287]
#cfg.MODEL.PIXEL_STD = [31.49143693, 33.6597322, 35.81154082]
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TRAIN = 600
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 600

predictor = DefaultPredictor(cfg)


data_out = []
for img in data:
    print(img["file_name"])
    im = cv2.imread(img["file_name"])
    outputs = predictor(im)
    instances = outputs["instances"]
    classes = instances.get("pred_classes").cpu().numpy().astype(int)
    scores = instances.get("scores").cpu().numpy()
    boxes = instances.get("pred_boxes").tensor.cpu().numpy()

    # masks = instances.get("pred_masks").cpu().numpy()
    # if masks.shape[0] > 0:
    #     mask = masks[0, :, :].astype(np.uint8)
    # else:
    #     mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    
    # iou = IoU(mask.astype(float), gt_mask.astype(float))
    # print(iou)
    bboxes = []
    confidences = []
    for bbox, score in zip(boxes, scores):
        bboxes.append(bbox.tolist())
        confidences.append(float(score))

    data = {}
    data["transform"] = img["transform"]
    data["boxes"] = bboxes
    data["scores"] = confidences
    data["file_name"] = img["file_name"]
    data_out.append(data)

    if len(viz):
        v = Visualizer(im[:, :, ::-1], scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(viz, os.path.basename(img["file_name"])), out.get_image()[:, :, ::-1])

with open(output, "w") as fout:
    json.dump(data_out, fout)
    
