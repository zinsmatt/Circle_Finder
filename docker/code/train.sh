mkdir -p /wdata/train_images
mkdir -p /wdata/checkpoint

python3 -W ignore pansharpen_images.py $1  /wdata/train_images

python3 -W ignore generate_detectron_annotations.py train_all /wdata/train_images/dataset.json /wdata/train_annotations.json

# download pretrained backbone for initialization
wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl -O /wdata/pretrained_backbone.pkl

python3 -W ignore train_fasterRCNN.py /wdata/train_annotations.json /wdata/checkpoint/
