from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
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
parser.add_argument('--checkpoint', default="checkpoint/model_final.pth", help="Checkpoint")
parser.add_argument('--viz', default="", help="Vizualize prediction images")
args = parser.parse_args(sys.argv[1:])

print("Run inference")

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
checkpoint = args.checkpoint
print("Use model weights from ", checkpoint)
    

with open(dataset, "r") as fin:
    input_data = json.load(fin)



cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = checkpoint
SCORE_THRESHOLD = 0.5
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
cfg.MODEL.PIXEL_MEAN = [111.01797572, 102.54100801, 93.86145873]
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TRAIN = 600
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 600


predictor = DefaultPredictor(cfg)


data_out = []
idx = 1
for img in input_data:
    print(idx, "/", len(input_data), " : ", img["file_name"])
    im = cv2.imread(img["file_name"])
    outputs = predictor(im)
    instances = outputs["instances"]
    classes = instances.get("pred_classes").cpu().numpy().astype(int)
    scores = instances.get("scores").cpu().numpy()
    boxes = instances.get("pred_boxes").tensor.cpu().numpy()

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
    idx += 1

with open(output, "w") as fout:
    json.dump(data_out, fout)
    
