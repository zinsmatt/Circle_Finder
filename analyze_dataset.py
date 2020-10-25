#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:40:52 2020

@author: mzins
"""

import cv2
import numpy as np
import json

dataset_file = "/media/DATA1/Topcoder/circle_finder/pansharpen/labels_train.json"
dataset_file = "/media/DATA1/Topcoder/circle_finder/circle/labels_train.json"

with open(dataset_file, "r") as fin:
    data = json.load(fin)
    
means = []
stds = []
for i, f in enumerate(data):
    img = cv2.imread(f["file_name"]).reshape((-1, 3))
    means.append(np.mean(img, axis=0))
    stds.append(np.std(img, axis=0))
    print(i, "/", len(data))
means = np.vstack(means)
stds = np.vstack(stds)
print("mean = ", np.mean(means, axis=0))
print("std = ", np.mean(stds, axis=0))