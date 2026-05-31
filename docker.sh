#!/usr/bin/env bash
set -euo pipefail

IMAGE="${REAL_IMAGE:-real-release:latest}"
MODEL_DIR="${REAL_MODEL_DIR:-$HOME/.cache/huggingface}"

docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)":/workspace/app \
  -v "${MODEL_DIR}":/models \
  -e HF_HOME=/models/hf-cache \
  -e HONEST_LLAMA_CACHE_DIR=/models/hf-cache \
  -e HF_ENDPOINT \
  -e HF_TOKEN \
  -e OPENAI_API_KEY \
  "${IMAGE}"
