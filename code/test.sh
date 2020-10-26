#python pansharpen_images.py /media/DATA1/Topcoder/circle_finder/test  /media/DATA1/Topcoder/circle_finder/prod/test_images

python generate_detectron_annotations.py test /media/DATA1/Topcoder/circle_finder/prod/test_images/dataset.json temp/test_annotations.json

python inference.py temp/test_annotations.json temp/test_predictions.json --viz viz

python predictions_to_geojson.py temp/test_predictions.json solution