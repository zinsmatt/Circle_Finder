import rasterio
import fiona
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from shapely.geometry.polygon import Polygon
from math import *
import sys
import argparse

def compute_compactness(polygon):
    feature_geom = Polygon(polygon)
    feature = feature_geom.area
    unit_circle = feature_geom.length ** 2 / (4 * pi)
    compactness = feature / unit_circle
    return compactness

def bbox_to_circle(bbox, sampling=100):
    x1, y1, x2, y2 = bbox
    r = (x2 - x1) / 2
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    t = np.linspace(0, 2*np.pi, sampling)
    x = r * np.cos(t) + mx
    y = r * np.sin(t) + my
    return np.vstack((x, y)).T



parser = argparse.ArgumentParser(description="Pansharpen images")
parser.add_argument("input", help="Input predictions")
parser.add_argument("output",help="Ouput folder")
args = parser.parse_args(sys.argv[1:])


predictions_file = args.input
output_folder = args.output

# predictions_file = "/home/mzins/dev/detectron2/prediction_valid.json"
# output_folder = "solution"


split_file = "split_file.csv"



with open(predictions_file, "r") as fin:
    data = json.load(fin)

split_data = []
for index, pred in enumerate(data):
    # print(file)
    
    file = pred["file_name"]
    name = os.path.splitext(os.path.basename(file))[0].split("_")[0]
    
    T = np.asarray(pred["transform"])
    boxes = pred["boxes"]
    scores = pred["scores"]
    transformed_polygons = []
    for bbox, score in zip(boxes, scores):
        if score > 0.9:
            # pts = np.asarray(poly) * s
            # minx, miny = np.min(pts, axis=0)
            # maxx, maxy = np.max(pts, axis=0)
            # corners = np.array([[minx, miny],
            #                     [maxx, maxy]])
            corners = np.asarray(bbox).reshape((-1, 2))
            corners = (T[:2, :2] @ corners.T + T[:2, 2].reshape((-1, 1))).T
            pts = bbox_to_circle(corners.flatten().tolist())
    
            # c = compute_compactness(pts)
            # print(index, c)
            transformed_polygons.append(pts)

    
    out_name = name + "_anno.geojson"
    out_file = os.path.join(output_folder, out_name)
    out_data = {"type": "FeatureCollection",
                "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::32619" } }}
    
    features = []
    print(len(transformed_polygons))
    for p in transformed_polygons:
        geometry = {}
        geometry["coordinates"] = [p.tolist()]
        feature = {"geometry": geometry}
        features.append(feature)

    out_data["features"] = features
    
    with open(out_file, "w") as fout:
        json.dump(out_data, fout)

    split_data.append(",".join([str(index), out_name, "provisional"]) + "\n")
    
with open(split_file, "w") as fout:
    fout.writelines(split_data)
