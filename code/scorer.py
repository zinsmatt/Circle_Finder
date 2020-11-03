import geojson, sys
from shapely.geometry.polygon import Polygon
from math import *

CIRCULAR_THRES = 0.85
IOU_THRES = 0.5

def load_polygons(geo_json_file):
    ret = []
    try:
        with open(geo_json_file) as f:
            data = geojson.load(f)['features']
            for i in range(len(data)):
                polygons = data[i]['geometry']['coordinates']
                for polygon in polygons:
                    feature_geom = Polygon(polygon)
                    feature = feature_geom.area
                    unit_circle = feature_geom.length ** 2 / (4 * pi)
                    compactness = feature / unit_circle
                    if feature_geom.is_valid and compactness >= CIRCULAR_THRES:
                        ret.append(feature_geom)
    except:
        pass

    return ret

def get_IoU(poly1, poly2):
    i = poly1.intersection(poly2).area
    u = poly1.area + poly2.area - i
    return i / u


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('[usage] <phase> <truth dir> <submission dir>')
        print('        <phase> should be train, provisional, or final')
        sys.exit(-1)
    SUBMISSION_DIR = sys.argv[3]
    TRUTH_DIR = sys.argv[2]
    PHASE = sys.argv[1]


    # SUBMISSION_DIR = "."
    # TRUTH_DIR = "/media/DATA1/Topcoder/circle_finder/gt_annotations"
    # PHASE = "provisional"
    SPLIT_FILE = "split_file.csv"

    target = set()
    for line in open(SPLIT_FILE):
        parts = line.strip().split(',')
        filename = parts[1]
        phase = parts[2]
        if phase == PHASE:
            target.add(filename)
    print(len(target))


    score = 0
    for filename in target:
        # pred = load_polygons(SUBMISSION_DIR + '/solution/' + filename)
        pred = load_polygons(SUBMISSION_DIR + "/" + filename)
        truth = load_polygons(TRUTH_DIR + '/' + filename)
        if len(truth) == 0:
            f1 = float(len(pred) == 0)
        elif len(pred) == 0:
            f1 = 0
        elif len(pred) > 2000:
            f1 = 0
        else:
            truth_area = [x.area for x in truth]
            matched = [False for i in range(len(truth))]
            overlap = 0
            for pred_poly in pred:
                best_IoU, best_i = IOU_THRES, -1
                pred_poly_area = pred_poly.area
                for i in range(len(truth)):
                    if not matched[i]:
                        max_IoU = min(pred_poly_area, truth_area[i]) / max(pred_poly_area, truth_area[i])
                        if max_IoU > best_IoU:
                            cur_IoU = get_IoU(pred_poly, truth[i])
                            if cur_IoU > best_IoU:
                                best_IoU, best_i = cur_IoU, i
                if best_IoU > IOU_THRES:
                    matched[best_i] = True
                    overlap += 1
                    if overlap == len(truth):
                        break

            precision = overlap / len(pred)
            recall = overlap / len(truth)
            if overlap == 0:
                f1 = 0
            else:
                f1 = precision * recall * 2 / (precision + recall)
                print(f1)
        score += f1
    score /= len(target)
    print('Final Score =', score)