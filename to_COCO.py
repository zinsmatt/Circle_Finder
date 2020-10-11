#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:33:56 2020

@author: mzins
"""

import json


filename = "labels.json"

with open(filename, "r") as fin:
    data = json.load(fin)
    
    
data_out = {}
images = {}

images = []
for f in data:
    img = {}
    img["file_name"] = f["file_name"]
    img["height"] = f["height"]
    img["width"] = f["width"]
    img["id"] = f["id"]
    images.append(img)
    
annotations = []
idx = 0
for image_id, f in enumerate(data):

    for a in f["annotations"]:
        annot = {}
        annot["id"] = str(idx)
        annot["image_id"] = image_id
        annot["category_id"] = 1#a["category_id"]
        annot["segmentation"] = a["segmentation"]
        bbox = a["bbox"]
        annot["bbox"] = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        annot["is_crowd"] = 0
        annotations.append(annot)
        idx += 1
        
        
categories = [{
            "supercategory": None,
            "id": 0,
            "name": "_background_"
        },
        {
            "supercategory": None,
            "id": 1,
            "name": "aeroplane"
        }
        ]
data_out["images"] = images
data_out["annotations"] = annotations
data_out["categories"] = categories


with open("coco_labels.json", "w") as fout:
    json.dump(data_out, fout)