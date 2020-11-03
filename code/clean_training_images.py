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
training_data = "temp/valid_annotations.json"

# training_data = "/media/DATA1/Topcoder/circle_finder/circle/labels_train.json"
# training_data = "/home/mzins/dev/Circle_Finder/detectron_labels_train.json"


# BAD_FILE = "clean/train/BAD.txt"
# GOOD_FILE = "clean/train/GOOD.txt"

BAD_FILE = "clean/valid/BAD.txt"
GOOD_FILE = "clean/valid/GOOD.txt"

def my_dataset():
    with open(training_data, "r") as fin:
        data = json.load(fin)
    return data

with open(BAD_FILE, "r") as fin:
    lines = fin.readlines()
BAD = [x.strip() for x in lines]
with open(GOOD_FILE, "r") as fin:
    lines = fin.readlines()
GOOD = [x.strip() for x in lines]



DatasetCatalog.register("custom_dataset", my_dataset)
metadata = MetadataCatalog.get("custom_dataset")

dataset_dicts = my_dataset()
cv2.namedWindow("fen", cv2.WINDOW_NORMAL)



def save():
    with open(BAD_FILE, "w") as fout:
        for l in BAD:
            fout.write(l + "\n")

    with open(GOOD_FILE, "w") as fout:
        for l in GOOD:
            fout.write(l + "\n")



for d in dataset_dicts:
    if d["file_name"] in BAD or d["file_name"] in GOOD: continue

    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("fen", out.get_image()[:, :, ::-1])
    k = cv2.waitKey(-1)
    if k == ord("e"):
        break
    if k == ord('b'):
        print(d["file_name"])
        BAD.append(d["file_name"])
    else:
        GOOD.append(d["file_name"])
    save()

save()

