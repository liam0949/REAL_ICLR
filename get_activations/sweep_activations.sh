#!/usr/bin/env bash
set -euo pipefail

# Submit Slurm feature-extraction jobs for the REAL release datasets.
#
# Usage:
#   bash get_activations/sweep_activations.sh
#   bash get_activations/sweep_activations.sh llama2_7B Qwen2.5-7B
#
# Optional environment variables:
#   LOG_DIR=get_activations/logs
#   DEVICE=0
#   CONDA_ENV=your_env
#   HONEST_LLAMA_FEATURES_DIR=<FEATURES_DIR>
#   HONEST_LLAMA_MODEL_REGISTRY=<MODEL_REGISTRY_JSON>
#   HF_ENDPOINT=<optional_hf_endpoint>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${LOG_DIR:-get_activations/logs}"
DEVICE="${DEVICE:-0}"
mkdir -p "${LOG_DIR}"

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
  for dataset in tqa_mc2 tqa_gen_end_q; do
    job_name="get_activations_${model_name}_${dataset}"
    sbatch \
      --job-name="${job_name}" \
      --output="${LOG_DIR}/${job_name}.out" \
      --error="${LOG_DIR}/${job_name}.err" \
      get_activations/run_job_get_activations.sh \
        --model_name "${model_name}" \
        --dataset_name "${dataset}" \
        --device "${DEVICE}"
  done
done
