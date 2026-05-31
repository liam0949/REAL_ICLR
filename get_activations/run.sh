#!/usr/bin/env bash
set -euo pipefail

# Local multi-model feature extraction for the REAL release.
#
# Usage:
#   bash get_activations/run.sh
#   bash get_activations/run.sh llama2_7B Qwen2.5-7B
#
# Optional environment variables:
#   DEVICE=0
#   HONEST_LLAMA_FEATURES_DIR=<FEATURES_DIR>
#   HONEST_LLAMA_MODEL_REGISTRY=<MODEL_REGISTRY_JSON>
#   HF_ENDPOINT=<optional_hf_endpoint>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-0}"

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    "llama2_7B"
    "llama2-chat-7B"
    "Llama-2-13b-chat-hf"
    "Qwen2.5-7B"
    "Qwen2.5-7B-Instruct"
  )
fi

for model_name in "${MODELS[@]}"; do
  bash get_activations/get_activations.sh \
    --model_name "${model_name}" \
    --device "${DEVICE}"
done
