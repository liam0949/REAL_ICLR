#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --mem=200G
#
# Uncomment these for your cluster if required:
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail

# Slurm template for a single validate_2fold run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/validation"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

MODEL_NAME=""
MODEL_PREFIX=""
NUM_HEADS="48"
ALPHA="15"
INSTRUCTION_PROMPT="${INSTRUCTION_PROMPT:-default}"
JUDGE_NAME="${JUDGE_NAME:-<JUDGE_MODEL_NAME>}"
INFO_NAME="${INFO_NAME:-<INFO_MODEL_NAME>}"
SEED="42"
DEVICE="${DEVICE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --model_prefix)
      MODEL_PREFIX="$2"
      shift 2
      ;;
    --num_heads)
      NUM_HEADS="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --instruction_prompt)
      INSTRUCTION_PROMPT="$2"
      shift 2
      ;;
    --judge_name)
      JUDGE_NAME="$2"
      shift 2
      ;;
    --info_name)
      INFO_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: $0 --model_name MODEL [--num_heads 48 --alpha 15 --seed 42]" >&2
  exit 2
fi

cmd=(
  python validate_2fold.py
  --model_name "${MODEL_NAME}"
  --method REAL
  --num_heads "${NUM_HEADS}"
  --alpha "${ALPHA}"
  --codebook 32
  --units 8
  --per_ratio 0.0
  --instruction_prompt "${INSTRUCTION_PROMPT}"
  --device "${DEVICE}"
  --num_fold 2
  --use_center_of_mass
  --judge_name "${JUDGE_NAME}"
  --info_name "${INFO_NAME}"
  --seed "${SEED}"
)

if [[ -n "${MODEL_PREFIX}" ]]; then
  cmd+=(--model_prefix "${MODEL_PREFIX}")
fi

"${cmd[@]}"
