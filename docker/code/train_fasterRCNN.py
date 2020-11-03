from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import random
import argparse
import sys
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os


parser = argparse.ArgumentParser(description="Pansharpen images")
parser.add_argument("input", help="Input dataset file")
parser.add_argument("output", help="Output checkpoint")
args = parser.parse_args(sys.argv[1:])


training_data = args.input
output_folder = args.output

print("Training")


# a few images images were ignored during training because they seemed strange
BAD = ["1c7d08acf1268ee61392a76bb3c9cf51_PANSHARPEN.tif",
"6cc0b10d0f2a5c792ee5d84ece52b1b1_PANSHARPEN.tif",
"17ae5c2a537cb4bbb6b9799ad4ccbd91_PANSHARPEN.tif",
"dc19cbf05504f4df2b92f1df4d38ca20_PANSHARPEN.tif",
"5559cddfe2f00ecde70a264ae2016fb0_PANSHARPEN.tif",
"3dd960274e125dbd64a6ce3fc3bab06f_PANSHARPEN.tif",
"b5ad122382b0121868f89262ea59a5e5_PANSHARPEN.tif",
"a83f095c7b5a69174ab0d29d3b9cfa64_PANSHARPEN.tif",
"84e535cb81d1c5294b0f76de15edfa18_PANSHARPEN.tif",
"e944ba762190c66444bd657b13809275_PANSHARPEN.tif"]


def my_dataset():
    with open(training_data, "r") as fin:
        data = json.load(fin)
    filt_data = []
    for d in data:
        if os.path.basename(d["file_name"]) not in BAD:
            filt_data.append(d)
    return filt_data

DatasetCatalog.register("custom_dataset", my_dataset)
metadata = MetadataCatalog.get("custom_dataset")


cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("custom_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = "/wdata/pretrained_backbone.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0015  # pick a good LR
cfg.SOLVER.MAX_ITER = 20000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TRAIN = 600
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 600
cfg.OUTPUT_DIR = output_folder
cfg.INPUT.CROP.ENABLED = False
cfg.INPUT.RANDOM_FLIP = "none"

cfg.MODEL.PIXEL_MEAN = [111.01797572, 102.54100801, 93.86145873]

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()