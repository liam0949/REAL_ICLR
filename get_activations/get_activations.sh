#!/usr/bin/env bash
set -euo pipefail

# Single-model TruthfulQA feature extraction for the REAL release.
#
# Usage:
#   bash get_activations/get_activations.sh --model_name llama2_7B --device 0
#
# Extra arguments are forwarded to get_activations.py, for example:
#   --model_path <MODEL_DIR> --features_dir <FEATURES_DIR>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-llama2_7B}"
DEVICE="${DEVICE:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

for dataset in tqa_mc2 tqa_gen_end_q; do
  python get_activations/get_activations.py \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${dataset}" \
    --device "${DEVICE}" \
    --require_hf \
    "${EXTRA_ARGS[@]}"
done
