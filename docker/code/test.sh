mkdir -p /wdata/test_images
mkdir -p /wdata/checkpoint
mkdir -p $2

if [ ! -f /wdata/checkpoint/model_final.pth ]; then
    echo "Use home trained model."
    wget https://www.dropbox.com/s/ham5s9kq4yzpxx2/model_final_20000_clean_train_all.pth?dl=0 -O /wdata/checkpoint/home_trained_model.pth
    ckpt=/wdata/checkpoint/home_trained_model.pth
else
    echo "Use trained model."
    ckpt=/wdata/checkpoint/model_final.pth
fi


python3 pansharpen_images.py $1  /wdata/test_images


python3 -W ignore generate_detectron_annotations.py test /wdata/test_images/dataset.json /wdata/test_annotations.json


python3 -W ignore inference.py /wdata/test_annotations.json /wdata/test_predictions.json --checkpoint $ckpt


python3 -W ignore predictions_to_geojson.py /wdata/test_predictions.json $2