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
from detectron2.config import get_cfg
import os
import cv2
import rasterio

parser = argparse.ArgumentParser(description="Pansharpen images")
parser.add_argument("input", help="Input dataset file")
parser.add_argument("output", help="Output checkpoint")
args = parser.parse_args(sys.argv[1:])


training_data = args.input
output_folder = args.output

# training_data = "/media/DATA1/Topcoder/circle_finder/pansharpen/labels_train.json"
# training_data = "/media/DATA1/Topcoder/circle_finder/circle/labels_train.json"
# training_data = "/home/mzins/dev/Circle_Finder/detectron_labels_train.json"


def my_dataset():
    with open(training_data, "r") as fin:
        data = json.load(fin)
    return data

DatasetCatalog.register("custom_dataset", my_dataset)
metadata = MetadataCatalog.get("custom_dataset")

# dataset_dicts = my_dataset()
# cv2.namedWindow("fen", cv2.WINDOW_NORMAL)
# for d in random.sample(dataset_dicts, 50):
#     # img = cv2.imread(d["file_name"])
#     with rasterio.open(d["file_name"]) as src:
#         img = src.read()
#     img = img.transpose((1, 2, 0))
#     img = img[:, :, :3]
#     visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("fen", out.get_image()[:, :, ::-1])
#     cv2.waitKey(-1)



cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("custom_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = "/home/mzins/dev/Circle_Finder/code/pretrained/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.MODEL.WEIGHTS = "/home/mzins/dev/Circle_Finder/code/pretrained/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.MODEL.WEIGHTS = "checkpoints/model_final.pth"
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_cpu.pth")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.BASE_LR = 0.00015  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
# cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TRAIN = 600
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 600
#cfg.INPUT.CROP.ENABLED = True
cfg.OUTPUT_DIR = output_folder

# mean =  [ 92.7146011   93.66882446 102.28063327]
# std =  [31.49143693 33.6597322  35.81154082]
# cfg.MODEL.PIXEL_MEAN = [102.28063327, 93.66882446, 92.7146011]
cfg.MODEL.PIXEL_MEAN = [111.01797572, 102.54100801, 93.86145873]
# cfg.MODEL.PIXEL_MEAN = [116.05156287, 116.05156287, 116.05156287]
#cfg.MODEL.PIXEL_STD = [31.49143693, 33.6597322, 35.81154082]
# cfg.MODEL.PIXEL_STD = [31.49143693, 33.6597322, 35.81154082, 30, 30, 30, 30, 30]

cfg.MODEL.PIXEL_MEAN = [92.96026726, 94.07466555, 102.3823547, 110.66299919, 110.32333314, 122.28574999, 138.12868997, 138.4608601]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader   # the default mapper


class MyCustomResize(T.Augmentation):
    def get_transform(self, image):
        old_h, old_w = image.shape[:2]
        f = 800 / old_w
        new_h, new_w = int(f * old_h), int(f * old_w)
        return T.ResizeTransform(old_h, old_w, new_h, new_w)


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)

    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=[
            # MyCustomResize()
            # T.RandomCrop("relative_range", [0.9, 0.9]),
            # T.RandomFlip(0.5, horizontal=True,  vertical=False),
            # T.RandomFlip(0.5, horizontal=False, vertical=True)
        ]))
        return dataloader



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg) 
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()