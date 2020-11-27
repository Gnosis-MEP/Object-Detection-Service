FROM tensorflow/tensorflow:1.12.3-gpu-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y \
        libxrender1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install numpy==1.16.4 opencv-python-headless==4.0.0.21