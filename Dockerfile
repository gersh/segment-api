ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx diffusers runpod
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth --directory-prefix /src
COPY . /src

CMD python -u /src/runpod_handler.py