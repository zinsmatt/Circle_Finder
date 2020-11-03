#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:33:56 2020

@author: mzins
"""

import json
import numpy as np

filename = "labels.json"

with open(filename, "r") as fin:
    data = json.load(fin)
    
out = {}

lines = []
for f in data:
   
    annotations = {}
    
    for i, a in enumerate(f["annotations"]):
        pts = np.asarray(a["segmentation"][0]).reshape((-1, 2))
        shape = {"name":"polygon"}
        shape["all_points_x"] = pts[:, 0].tolist()
        shape["all_points_y"] = pts[:, 1].tolist()
        annotations[i] = {"shape_attributes" : shape}
   
    img_data = {}
    img_data["filename"] = f["file_name"]
    img_data["regions"] = annotations
    out[f["file_name"]] = img_data

with open("circle/train/via_region_data.json", "w") as fout:
    json.dump(out, fout)
    
    
    