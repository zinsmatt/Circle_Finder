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



def create_if_not(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)



def clean_poly(pts):
    out = [pts[0]]
    for i in range(1, len(pts)):
        if np.sum((pts[i, :] - pts[i-1, :])**2) > 1e-3:
            out.append(pts[i, :])
    return np.vstack(out)

mode = "valid"

print("Mode: ", mode)

dataset_folder = "/media/DATA1/Topcoder/circle_finder/"

if mode == "train" or mode == "valid":
    folders = glob.glob(os.path.join(dataset_folder, "train/*"))
else:
    folders = glob.glob(os.path.join(dataset_folder, "test/*"))


inputs = []
for f in folders:
    img_file = os.path.join(f, os.path.basename(f) + "_PAN.tif")
    if mode == "train" or mode == "valid":
        annot_file = os.path.join(f, os.path.basename(f) + "_anno.geojson")
    else:
        annot_file = ""
    inputs.append((img_file, annot_file))



labelled_data = []
if mode == "train":
    list_indices = sorted(list(range(0, len(inputs), 3)) + list(range(1, len(inputs), 3)))
elif mode == "valid":
    list_indices = range(2, len(inputs), 3)
else:
    list_indices = range(len(inputs))

out_folder = os.path.join(dataset_folder, "circle/%s/" % mode)
create_if_not(out_folder)

for idx in list_indices:
    print(idx)
    name = os.path.basename(inputs[idx][0]).split('_')[0]

    filename = os.path.join(out_folder, name + ".png")
    f_annotation = inputs[idx][1]
    f_pan = inputs[idx][0]

    with rasterio.open(f_pan) as src:
        img = src.read(1).astype(float)
        transform= src.transform

    img = np.dstack([img]*3)
    pil_img = Image.fromarray(img.astype(np.uint8))

    x_scale = 1
    y_scale = 1
    if img.shape[1] > 800:
        x_scale = 800 / img.shape[1]
        y_scale = x_scale
        new_width = int(round(x_scale * img.shape[1]))
        new_height = int(round(y_scale * img.shape[0]))
        pil_img = pil_img.resize((new_width, new_height))

    pil_img.save(filename)
    w, h = pil_img.size

    annotations = []
    if mode == "train" or mode == "valid":
        with fiona.open(f_annotation, "r") as annotation_collection:
            annotations = [feature["geometry"] for feature in annotation_collection]

        with rasterio.open(f_pan) as src:
            out_image, out_transform = rasterio.mask.mask(src, annotations, all_touched=False, invert=False, crop=False)
            out_meta = src.meta

        mask = out_image.squeeze()
        mask[mask!=0] = 1


        polys = [np.vstack(a["coordinates"]) for a in annotations]

        transform = np.asarray(transform).reshape((3, 3))
        t_inv = np.linalg.inv(transform)
        annotations = []
        for pts in polys:

            pts= pts[:, :2]
            pts = (t_inv @ np.vstack((pts.T, np.ones((1, pts.shape[0]))))).T
            uvs = np.round(pts[:, :2])
            uvs[:, 0] *= x_scale
            uvs[:, 1] *= y_scale
            xmin, ymin = np.min(uvs, axis=0)
            xmax, ymax = np.max(uvs, axis=0)

            uvs = clean_poly(uvs)

            annot = {}
            annot["bbox"] = [int(xmin), int(ymin), int(xmax), int(ymax)]
            annot["bbox_mode"] = 0
            annot["category_id"] = 0
            annot["segmentation"] = [uvs.flatten().tolist()]
            annotations += [annot]


    data = {"file_name": filename,
            "height": h,
            "width": w,
            "id": idx,
            "annotations":annotations,
            "transform":transform.tolist(),
            "scaling": 1 / x_scale}

    labelled_data.append(data)




with open(os.path.join(dataset_folder, "circle", "labels_%s.json" % mode), "w") as fout:
    json.dump(labelled_data, fout)
