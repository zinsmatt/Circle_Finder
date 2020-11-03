import json
import os



dataset_file = "/media/DATA1/Topcoder/circle_finder/wdata/test_images/dataset.json"
out_file = "/media/DATA1/Topcoder/circle_finder/wdata/test_dataset.json"
images_path = "/wdata/test_images/"
annotations_path = "/data/test/"

# dataset_file = "/media/DATA1/Topcoder/circle_finder/wdata/train_images/dataset.json"
# out_file = "/media/DATA1/Topcoder/circle_finder/wdata/train_dataset.json"
# images_path = "/wdata/train_images/"
# annotations_path = "/data/train/"

def get_f(x):
    xx = x.split("/")
    print(xx)
    return os.path.join(xx[-2], xx[-1])


with open(dataset_file, "r") as fin:
    dataset = json.load(fin)
    
out_data = []
print(dataset)
for data in dataset:
    a = data["image"]
    b = data["annotation"]
    aa = os.path.join(images_path, os.path.basename(a))
    bb = os.path.join(annotations_path, get_f(b))
    d = {"image":aa, "annotation":bb}
    out_data.append(d)
with open(out_file, "w") as fout:
    json.dump(out_data, fout)