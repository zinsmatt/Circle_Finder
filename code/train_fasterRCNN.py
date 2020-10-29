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
# for d in random.sample(dataset_dicts, 5):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("fen", out.get_image()[:, :, ::-1])
#     cv2.waitKey(-1)



cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("custom_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4

# cfg.MODEL.WEIGHTS = "/home/mzins/dev/Circle_Finder/code/pretrained/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.MODEL.WEIGHTS = "/home/mzins/dev/Circle_Finder/code/pretrained/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_cpu.pth")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
# cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TRAIN = 600
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 600
cfg.OUTPUT_DIR = output_folder
cfg.INPUT.CROP.ENABLED = True
# mean =  [ 92.7146011   93.66882446 102.28063327]
# std =  [31.49143693 33.6597322  35.81154082]
# cfg.MODEL.PIXEL_MEAN = [102.28063327, 93.66882446, 92.7146011]
cfg.MODEL.PIXEL_MEAN = [111.01797572, 102.54100801, 93.86145873]
# cfg.MODEL.PIXEL_MEAN = [116.05156287, 116.05156287, 116.05156287]
# cfg.MODEL.PIXEL_STD = [31.49143693, 33.6597322, 35.81154082]


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()