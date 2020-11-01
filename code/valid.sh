#python pansharpen_images.py /media/DATA1/Topcoder/circle_finder/train  /media/DATA1/Topcoder/circle_finder/prod/train_images

#python generate_detectron_annotations.py valid /media/DATA1/Topcoder/circle_finder/prod/train_images/dataset.json temp/valid_annotations.json


#python inference.py temp/valid_annotations.json temp/valid_predictions.json 

python predictions_to_geojson.py temp/valid_predictions.json solution

python scorer.py provisional /media/DATA1/Topcoder/circle_finder/gt_annotations solution/