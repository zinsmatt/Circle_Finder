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
ENV PATH /opt/conda/bin:$PATH



# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     bash ~/miniconda.sh -b -p $HOME/miniconda

# RUN export PATH="$HOME/miniconda/bin:$PATH" &&  conda create -n detectron2 python=3.6 && source activate detectron2 && conda install -c pytorch pytorch torchvision -y


# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda clean -tipsy && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc


RUN mkdir build && \
    cd build && \
    git clone https://github.com/facebookresearch/detectron2.git


# RUN export PATH="$HOME/miniconda/bin:$PATH" && conda activate detectron2 && python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html 
# RUN export PATH="$HOME/miniconda/bin:$PATH" && conda activate detectron2 && pip3 install opencv-python

RUN pip3 install torch torchvision opencv-python
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html 


COPY . .

# RUN cd build/detectron2 && \
#     cd demo && \
#     wget https://miro.medium.com/max/700/1*EYFejGUjvjPcc4PZTwoufw.jpeg && \
#     mv '1*EYFejGUjvjPcc4PZTwoufw.jpeg' test.jpeg && \
#     python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#     --input test.jpeg \
#     --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
RUN cp demo2.py build/detectron2/demo
ENTRYPOINT  ["python3", "run_pipeline.py"]
