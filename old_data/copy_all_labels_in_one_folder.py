#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:29:50 2020

@author: mzins
"""
import os
import shutil
import glob

in_folder = "/media/DATA1/Topcoder/circle_finder/train"
out_folder = "/media/DATA1/Topcoder/circle_finder/gt_annotations"


folders = glob.glob(os.path.join(in_folder, "*"))


for f in folders:
    label_file = os.path.join(f, os.path.basename(f) + "_anno.geojson")
    out = os.path.join(out_folder, os.path.basename(label_file))
    shutil.copy(label_file, out)
