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
# For this exercise, the files are collocated with the notebook. 




folders = glob.glob("/media/DATA1/Topcoder/circle_finder/train/*")

inputs = []
for f in folders:
    img_file = os.path.join(f, os.path.basename(f) + "_PAN.tif")
    annot_file = os.path.join(f, os.path.basename(f) + "_anno.geojson")
    inputs.append((img_file, annot_file))



def clean_poly(pts):
    out = [pts[0]]
    for i in range(1, len(pts)):
        if np.sum((pts[i, :] - pts[i-1, :])**2) > 1e-3:
            out.append(pts[i, :])
    return np.vstack(out)

labelled_data = []
idx = 45
for idx in range(40):#len(inputs)):
    print(idx)
    f_annotation = inputs[idx][1]
    f_pan = inputs[idx][0]

    darknet_lines = []

    with fiona.open(f_annotation, "r") as annotation_collection:
        annotations = [feature["geometry"] for feature in annotation_collection]
                        

    with rasterio.open(f_pan) as src:
        out_image, out_transform = rasterio.mask.mask(src, annotations, all_touched=False, invert=False, crop=False)
        out_meta = src.meta

        img = src.read(1).astype(float)
        mask = out_image.squeeze()
        mask[mask!=0] = 1

        img = np.dstack([img]*3)
        # img[:, :, 0] += 100*mask.astype(float)
        # img = np.clip(img, 0, 255)
        
        pil_img = Image.fromarray(img.astype(np.uint8))
        filename = "/home/mzins/dev/Circle_Finder/circle/train/img_%06d.png" % idx
        # filename = "/home/mzins/dev/darknet/build/darknet/x64/data/obj/img_%06d.png" % idx
        # x_scale = 416/ img.shape[1]
        # y_scale = 416/ img.shape[0]
        x_scale = 1
        y_scale = 1
        
        # pil_img.resize((416, 416)).save(filename)
        pil_img.save(filename)
        #plt.imshow(img/255)
        #plt.show()


        # poly = rasterio.features.geometry_mask(geometries=annotations, out_shape=(src.height, src.width), transform=src.transform, all_touched=False, invert=True)
        # plt.imshow(poly)
        # plt.show()
        polys = [np.vstack(a["coordinates"]) for a in annotations]

        
        transform = np.asarray(src.transform).reshape((3, 3))
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


            darknet_lines.append("0 %f %f %f %f\n" % (xmin / 416, ymin / 416, (xmax-xmin) / 416, (ymax-ymin) / 416))


    data = {"file_name": filename,
            "height": img.shape[0],
            "width": img.shape[1],
            "id": idx,
            "annotations":annotations}
    

    labelled_data.append(data)

    with open(os.path.splitext(filename)[0] + ".txt", "w") as fout:
        fout.writelines(darknet_lines)


with open("labels.json", "w") as fout:
    json.dump(labelled_data, fout)
# out_meta.update({"driver": "GTiff",
#                  "height": out_image.shape[1],
#                  "width": out_image.shape[2],
#                  "transform": out_transform})



# with rasterio.open("Masked.tif", "w", **out_meta) as dest:
#     dest.write(out_image)


# In[ ]:




