#!/usr/bin/env bash
#SBATCH --job-name=get_activations
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --mem=100G
# Uncomment these for your cluster if required:
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail

# Slurm template for extracting one feature bank.
#
# Example:
#   sbatch get_activations/run_job_get_activations.sh \
#     --model_name llama2_7B \
#     --dataset_name tqa_mc2 \
#     --device 0
#
# Optional environment variables:
#   CONDA_ENV=your_env
#   HONEST_LLAMA_FEATURES_DIR=<FEATURES_DIR>
#   HONEST_LLAMA_MODEL_REGISTRY=<MODEL_REGISTRY_JSON>
#   HF_ENDPOINT=<optional_hf_endpoint>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

MODEL_NAME=""
DATASET_NAME=""
DEVICE="${DEVICE:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
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

if [[ -z "${MODEL_NAME}" || -z "${DATASET_NAME}" ]]; then
  echo "Usage: $0 --model_name MODEL --dataset_name DATASET [--device 0]" >&2
  exit 2
fi

python get_activations/get_activations.py \
  --model_name "${MODEL_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --device "${DEVICE}" \
  --require_hf \
  "${EXTRA_ARGS[@]}"
