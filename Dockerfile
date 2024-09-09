# docker build --tag=sebsti1/pfas .
# docker run -it -p 127.0.0.1:8080:8080 -v ./src:/root/src --rm sebsti1/pfas /bin/bash
FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

# install wget and packages from here https://stackoverflow.com/questions/55313610
RUN apt update &&\
    apt install -y wget ffmpeg libsm6 libxext6  -y

RUN mkdir -p ~/miniconda3 &&\
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh &&\
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 &&\
    rm ~/miniconda3/miniconda.sh

ENV PATH=/root/miniconda3/bin:/root/miniconda3/condabin:$PATH

RUN conda init bash &&\
    conda create -y -n pfas python=3.10

RUN eval "$(conda shell.bash hook)" &&\
    conda activate pfas &&\
    pip install numpy matplotlib opencv-contrib-python imageio Cython Pillow scikit-image scipy six open3d imutils notebook &&\
    conda install -c conda-forge jupyterlab

RUN mkdir /root/src/ &&\
    echo $'eval "$(conda shell.bash hook)"\ncd /root/src\nconda activate pfas\njupyter lab --allow-root --ip=0.0.0.0 --port=8080' >> /entrypoint.sh &&\
    chmod +x entrypoint.sh

#EXPOSE 8080
#ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]