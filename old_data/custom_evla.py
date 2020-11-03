import json
import numpy as np
import cv2
from shapely.geometry.polygon import Polygon



def IoU(a, b):
    minax, minay = np.min(a, axis=0)
    maxax, maxay = np.max(a, axis=0)
    minbx, minby = np.min(b, axis=0)
    maxbx, maxby = np.max(b, axis=0)
    ok = True
    if minax > maxbx or minbx > maxax:
        ok = False
    if minay > maxby or minby > maxay:
        ok = False
    if not ok:
        return 0
    
    
    pad = 2
    minx = min(minax, minbx) - pad
    miny = min(minay, minby) - pad
    maxx = max(maxax, maxbx) + pad
    maxy = max(maxay, maxby) + pad
    
    w = maxx - minx
    h = maxy - miny
#    scale = 400 / w
    scale = 1
    a -= [minx, miny]
    b -= [minx, miny]

    a *= scale
    b *= scale
    a = np.round(a).astype(int)
    b = np.round(b).astype(int)
    ma = np.zeros((int(scale * h) + 1, int(scale * w) + 1), np.uint8)
    mb = np.zeros((int(scale * h) + 1, int(scale * w) + 1), np.uint8)
    cv2.drawContours(ma, [a], -1, 1, -1)
    cv2.drawContours(mb, [b], -1, 1, -1)
    
    # m = np.zeros((int(scale * h) + 1, int(scale * w) + 1), np.uint8)
    # cv2.drawContours(m, [b], -1, 150, -1)
    # cv2.drawContours(m, [a], -1, 255, -1)

    # cv2.imshow("ma", ma*255)
    # cv2.imshow("mb", mb*255)
    # cv2.imshow("m", m)
    # cv2.waitKey(-1)

    ma = ma.astype(float)
    mb = mb.astype(float)
    inter = np.sum(ma * mb)
    union = np.sum(ma + mb)
    iou = inter / (union - inter)
    return iou

def get_IoU(poly1, poly2):
    i = poly1.intersection(poly2).area
    u = poly1.area + poly2.area - i
    return i / u



def clean_poly(pts):
    out = [pts[0]]
    for i in range(1, len(pts)):
        if np.sum((pts[i, :] - out[-1])**2) > 1e-3:
            out.append(pts[i, :])
    return np.vstack(out)

valid_dataset = "/media/DATA1/Topcoder/circle_finder/circle/labels_valid.json"

pred_dataset = "/home/mzins/dev/detectron2/prediction_valid.json"

with open(valid_dataset, "r") as fin:
    gt = json.load(fin)

with open(pred_dataset, "r") as fin:
    pred = json.load(fin)


F1_scores = []
for file in gt:
    filename = file["file_name"]
    pred_polys = [Polygon(clean_poly(np.asarray(p))) for p in pred[filename]["polygons"]]

    gt_polys = [Polygon(clean_poly(np.asarray(annot["segmentation"]).flatten().reshape((-1, 2)))).convex_hull for annot in file["annotations"]]
    

       
    if len(pred_polys) > 2000:
        F1 = 0
    elif len(pred_polys) == 0 and len(gt_polys) > 0 or len(pred_polys) > 0 and len(gt_polys) == 0:
        F1 = 0
    else:
        done = [False] * len(gt_polys)
        iou = []    
        nb_matches = 0
        for i in range(len(pred_polys)):
            best_iou = 0
            best_iou_idx = -1
            for j in range(len(gt_polys)):
                if not done[j]:
                    iou = get_IoU(pred_polys[i], gt_polys[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_iou_idx = j
            if best_iou > 0.5:
                nb_matches += 1
                done[best_iou_idx] = True
        precision = nb_matches / len(pred_polys)
        recall = nb_matches / len(gt_polys)
        F1 = 0
        if precision + recall > 0:
            F1 = precision * recall * 2 / (precision + recall)
    F1_scores.append(F1)

print("Mean F1 score = %.3f" % (100 * np.mean(F1_scores)))