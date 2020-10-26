import os
import glob
import subprocess
import json
import time
import argparse
import sys

parser = argparse.ArgumentParser(description="Pansharpen images")
parser.add_argument("input", help="Input folder")
parser.add_argument("output",help="Destination folder")
args = parser.parse_args(sys.argv[1:])

# dataset_folder = "/media/DATA1/Topcoder/circle_finder/train/*"
# out_folder = "/media/DATA1/Topcoder/circle_finder/prod/train_images"
# dataset_folder = "/media/DATA1/Topcoder/circle_finder/test/*"
# out_folder = "/media/DATA1/Topcoder/circle_finder/prod/test_images"

dataset_folder = args.input
out_folder = args.output

folders = glob.glob(os.path.join(dataset_folder, "*"))

ta = time.time()
dataset = []
for f in folders:
    image_id = os.path.basename(f)
    
    pan_file = os.path.join(f, image_id + "_PAN.tif")
    mul_file = os.path.join(f, image_id + "_MUL.tif")
    out_file = os.path.join(out_folder, image_id + "_PANSHARPEN.tif")
    annotation = os.path.join(f, image_id + "_anno.geojson")

    cmd = ["gdal_pansharpen.py", pan_file, mul_file, out_file]#, "-b", "5", "-b", "3", "-b", "2"]
    subprocess.run(cmd)
    print(image_id)
    data = {"image": out_file, "annotation": annotation}
    dataset.append(data)
    
with open(os.path.join(out_folder, "dataset.json"), "w") as fout:
    json.dump(dataset, fout)
    
    
print("Pansharpening of %d image done in %.3fs" % (len(folders), time.time()-ta))