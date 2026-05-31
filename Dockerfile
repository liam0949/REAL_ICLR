ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    HONEST_LLAMA_CACHE_DIR=/models/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app

COPY requirements.txt ./requirements.txt
COPY TruthfulQA ./TruthfulQA

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja && \
    python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p \
    features \
    validation/results_dump/head_sorted \
    validation/results_dump/answer_dump \
    validation/results_dump/summary_dump \
    validation/splits \
    validation/sweeping/logs

CMD ["/bin/bash"]
