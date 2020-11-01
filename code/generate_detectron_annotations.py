#!/usr/bin/env python
# coding: utf-8


import rasterio
import rasterio.mask
import fiona
import os
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

import argparse
import sys

parser = argparse.ArgumentParser(description="Pansharpen images")
parser.add_argument("mode", help="Mode (can be train, valid or test)")
parser.add_argument("input", help="Input dataset file")
parser.add_argument("output",help="Ouput annotation file")
args = parser.parse_args(sys.argv[1:])

mode = args.mode
dataset_file = args.input
out_detectron_annotations = args.output




def clean_poly(pts):
    out = [pts[0]]
    for i in range(1, len(pts)):
        if np.sum((pts[i, :] - pts[i-1, :])**2) > 1e-3:
            out.append(pts[i, :])
    return np.vstack(out)


# mode = "test"
# out_detectron_annotations = "detectron_labels_" + mode + ".json"
# # dataset_file = "/media/DATA1/Topcoder/circle_finder/prod/train_images/dataset.json"
# dataset_file = "/media/DATA1/Topcoder/circle_finder/prod/test_images/dataset.json"


print("Mode: ", mode)
with open(dataset_file, "r") as fin:
    dataset = json.load(fin)
    

if mode == "train":
    list_indices = sorted(list(range(0, len(dataset), 3)) + list(range(1, len(dataset), 3)))
elif mode == "valid":
    list_indices = range(2, len(dataset), 3)
else:
    list_indices = range(len(dataset))



detectron_labels = []
for idx in list_indices:
    print(idx)
    image_file = dataset[idx]["image"]
    annotation_file = dataset[idx]["annotation"]


    with rasterio.open(image_file) as src:
        w = src.width
        h = src.height
        transform = np.asarray(src.transform).reshape((3, 3))

    annotations = []
    if mode == "train" or mode == "valid" or mode == "train_all":
        with fiona.open(annotation_file, "r") as annotation_collection:
            annotations = [feature["geometry"] for feature in annotation_collection]

        polys = [np.vstack(a["coordinates"]) for a in annotations]

        t_inv = np.linalg.inv(transform)
        annotations = []
        for pts in polys:
            pts = pts[:, :2]
            pts = (t_inv @ np.vstack((pts.T, np.ones((1, pts.shape[0]))))).T
            uvs = pts[:, :2]
            xmin, ymin = np.min(uvs, axis=0)
            xmax, ymax = np.max(uvs, axis=0)
            uvs = clean_poly(uvs)

            annot = {}
            annot["bbox"] = [int(xmin), int(ymin), int(xmax), int(ymax)]
            annot["bbox_mode"] = 0
            annot["category_id"] = 0
            annot["segmentation"] = [uvs.flatten().tolist()]

            annotations += [annot]


    data = {"file_name": image_file,
            "height": h,
            "width": w,
            "id": idx,
            "annotations":annotations,
            "transform":transform.tolist()}

    detectron_labels.append(data)




with open(out_detectron_annotations, "w") as fout:
    json.dump(detectron_labels, fout)
