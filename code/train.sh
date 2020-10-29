# python pansharpen_images.py /media/DATA1/Topcoder/circle_finder/train  /media/DATA1/Topcoder/circle_finder/prod/train_images

# python generate_detectron_annotations.py train /media/DATA1/Topcoder/circle_finder/prod/train_images/dataset.json temp/train_annotations.json
# python generate_detectron_annotations.py valid /media/DATA1/Topcoder/circle_finder/prod/train_images/dataset.json temp/valid_annotations.json


# python train_fasterRCNN.py temp/train_annotations.json checkpoint/
python train_fasterRCNN.py temp/train_all_annotations.json checkpoint/