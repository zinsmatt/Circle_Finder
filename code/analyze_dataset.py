import cv2
import numpy as np
import json
from PIL import Image
import argparse
import sys
import rasterio

parser = argparse.ArgumentParser(description="Analyze images")
parser.add_argument("input", help="Input dataset file")
args = parser.parse_args(sys.argv[1:])

dataset_file = args.input


with open(dataset_file, "r") as fin:
    data = json.load(fin)
    
means = []
stds = []
for i, f in enumerate(data[::10]):
    # img = cv2.imread(f["file_name"]).reshape((-1, 3))
    # img = np.asarray(Image.open(f["image"])).reshape((-1, 3))
    with rasterio.open(f["image"]) as src:
        img = src.read()
    img = img.reshape((img.shape[0], -1))

    # img = cv2.imread(f["file_name"]).reshape((-1, 3))
    means.append(np.mean(img, axis=1))
    stds.append(np.std(img, axis=1))
    
    print(i, "/", len(data))
means = np.vstack(means)
stds = np.vstack(stds)
print("mean = ", np.mean(means, axis=0))
print("std = ", np.mean(stds, axis=0))