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


BAD = ["/media/DATA1/Topcoder/circle_finder/prod/train_images/1c7d08acf1268ee61392a76bb3c9cf51_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/6cc0b10d0f2a5c792ee5d84ece52b1b1_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/17ae5c2a537cb4bbb6b9799ad4ccbd91_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/dc19cbf05504f4df2b92f1df4d38ca20_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/5559cddfe2f00ecde70a264ae2016fb0_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/3dd960274e125dbd64a6ce3fc3bab06f_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/b5ad122382b0121868f89262ea59a5e5_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/a83f095c7b5a69174ab0d29d3b9cfa64_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/84e535cb81d1c5294b0f76de15edfa18_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/70f7f473487c6c68cdeed74a8939828f_PANSHARPEN.tif",
"/media/DATA1/Topcoder/circle_finder/prod/train_images/e944ba762190c66444bd657b13809275_PANSHARPEN.tif"]


BAD = [os.path.basename(o) for o in BAD]

def my_dataset():
    with open(training_data, "r") as fin:
        data = json.load(fin)
    # filt_data = []
    # for d in data:
    #     if os.path.basename(d["file_name"]) not in BAD:
    #         filt_data.append(d)
    # print("############################################################## len data filt_data = ", len(data), len(filt_data))
    # return filt_data
    return data

DatasetCatalog.register("custom_dataset", my_dataset)
metadata = MetadataCatalog.get("custom_dataset")


# import cv2
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
# cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("custom_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4

# cfg.MODEL.WEIGHTS = "/home/mzins/dev/Circle_Finder/code/pretrained/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# cfg.MODEL.WEIGHTS = "/home/mzins/dev/Circle_Finder/code/pretrained/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_cpu.pth")
cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.BASE_LR = 0.0015  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TRAIN = 600
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 600
cfg.OUTPUT_DIR = output_folder
cfg.INPUT.CROP.ENABLED = False
cfg.INPUT.RANDOM_FLIP = "none"
# mean =  [ 92.7146011   93.66882446 102.28063327]
# std =  [31.49143693 33.6597322  35.81154082]
# cfg.MODEL.PIXEL_MEAN = [102.28063327, 93.66882446, 92.7146011]
#############cfg.MODEL.PIXEL_MEAN = [111.01797572, 102.54100801, 93.86145873]
cfg.MODEL.PIXEL_MEAN = [93.86145873, 102.54100801, 111.01797572]
# cfg.MODEL.PIXEL_MEAN = [116.05156287, 116.05156287, 116.05156287]
# cfg.MODEL.PIXEL_STD = [31.49143693, 33.6597322, 35.81154082]


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()