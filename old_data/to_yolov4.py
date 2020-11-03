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
    
    
lines = []
for f in data:
    img = {}
    img["file_name"] = f["file_name"]
    img["height"] = f["height"]
    img["width"] = f["width"]
    img["id"] = f["id"]
    
    l = f["file_name"] + " "
    for a in f["annotations"]:
        l += ",".join(map(str, a["bbox"])) + ",0 "
    
    lines.append(l + "\n")
with open("/home/mzins/dev/pytorch-YOLOv4/train.txt", "w") as fout:
    fout.writelines(lines)
    
    
    