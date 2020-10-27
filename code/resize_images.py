import numpy as np
import cv2
import rasterio
import os
import glob
import subprocess
import json
import time
import argparse
import sys
import shutil

parser = argparse.ArgumentParser(description="Resize images")
parser.add_argument("input", help="Input folder")
parser.add_argument("output",help="Destination folder")
args = parser.parse_args(sys.argv[1:])

dataset_folder = args.input
out_folder = args.output

with open(os.path.join(dataset_folder, "dataset.json"), "r") as fin:
    data = json.load(fin)


ta = time.time()
dataset = []
for idx, f in enumerate(data):
    img_file = f["image"]
    out_img_file = os.path.join(out_folder, os.path.basename(img_file))

    src = rasterio.open(img_file)
    w, h = src.width, src.height

    factor = 1.0
    if max(w, h) > 800:
        factor = 800 / max(w, h)
        new_w, new_h = int(factor * w), int(factor * h)
        with rasterio.open(img_file) as src:
            img = src.read()
            transform = src.transform
        img = np.vstack([np.expand_dims(cv2.resize(img[i, :, :], (new_w, new_h), interpolation=cv2.INTER_AREA), 0) for i in range(img.shape[0])])
        with rasterio.open(out_img_file, 'w', height=img.shape[1], width=img.shape[2], count=img.shape[0], dtype=str(img.dtype), transform=transform, driver="GTiff") as new_dataset:
            new_dataset.write(img)
            new_dataset.close()
    else:
        shutil.copy(img_file, out_img_file)

    print(idx, "/", len(data))
    out_data = {"image": out_img_file, "annotation": f["annotation"], "scale": factor}
    dataset.append(out_data)
    
with open(os.path.join(out_folder, "dataset.json"), "w") as fout:
    json.dump(dataset, fout)
    
print("Resizing of %d images done in %.3fs" % (len(data), time.time()-ta))