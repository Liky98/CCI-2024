FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        openmpi-bin \
        libopenmpi-dev \
        python3-dev \
        git && \ 
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && \
    pip install --no-deps trl peft accelerate bitsandbytes triton

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    rm /requirements.txt && \
    mkdir /CCI2024

COPY ./ /CCI2024

WORKDIR /CCI2024