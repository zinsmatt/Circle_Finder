from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import random
import argparse
import sys
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import cv2


# parser = argparse.ArgumentParser(description="Pansharpen images")
# parser.add_argument("input", help="Input dataset file")
# parser.add_argument("output", help="Output checkpoint")
# args = parser.parse_args(sys.argv[1:])


# training_data = args.input
# output_folder = args.output

training_data = "temp/train_annotations.json"
# training_data = "/media/DATA1/Topcoder/circle_finder/circle/labels_train.json"
# training_data = "/home/mzins/dev/Circle_Finder/detectron_labels_train.json"


def my_dataset():
    with open(training_data, "r") as fin:
        data = json.load(fin)
    return data

DatasetCatalog.register("custom_dataset", my_dataset)
metadata = MetadataCatalog.get("custom_dataset")

dataset_dicts = my_dataset()
cv2.namedWindow("fen", cv2.WINDOW_NORMAL)
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("fen", out.get_image()[:, :, ::-1])
    cv2.waitKey(-1)
