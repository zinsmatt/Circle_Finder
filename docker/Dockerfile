FROM ubuntu:focal

RUN \
  apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    curl \
    git \
    wget 

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
  
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN pip3 install torch torchvision opencv-python
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html 

RUN pip3 install rasterio fiona shapely
RUN apt-get update && apt-get install -y libgdal-dev
RUN apt-get install -y gdal-bin libgdal26

COPY . .

WORKDIR /code

#ENTRYPOINT  ["sh", "train.sh", "/data/train"]
ENTRYPOINT  ["sh", "test.sh", "/data/test", "/wdata/solution"]
