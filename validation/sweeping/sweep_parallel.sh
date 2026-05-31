#!/usr/bin/env bash
set -euo pipefail

# Slurm validation sweep template.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MAX_JOBS="${MAX_JOBS:-6}"
LOG_DIR="${LOG_DIR:-validation/sweeping/logs}"
INSTRUCTION_PROMPT="${INSTRUCTION_PROMPT:-default}"
JUDGE_NAME="${JUDGE_NAME:-<JUDGE_MODEL_NAME>}"
INFO_NAME="${INFO_NAME:-<INFO_MODEL_NAME>}"

mkdir -p "${LOG_DIR}"

if [[ $# -gt 0 ]]; then
  MODEL_NAMES=("$@")
else
  MODEL_NAMES=(
    "llama2_7B"
    "llama2-chat-7B"
    "Llama-2-13b-chat-hf"
    "Qwen2.5-7B"
    "Qwen2.5-7B-Instruct"
  )
fi

SEEDS=(42)

for model_name in "${MODEL_NAMES[@]}"; do
  for pair in "0 1" "15 48"; do
    alpha="${pair%% *}"
    num_heads="${pair##* }"
    for seed in "${SEEDS[@]}"; do
      while [[ "$(squeue -u "${USER}" | awk '$5 ~ /^(R|PD)$/' | wc -l)" -ge "${MAX_JOBS}" ]]; do
        echo "Maximum number of jobs (${MAX_JOBS}) reached. Waiting..."
        sleep 60
      done

      job_name="validate_2fold_${model_name}_a${alpha}_k${num_heads}_seed${seed}"
      sbatch \
        --job-name="${job_name}" \
        --output="${LOG_DIR}/${job_name}.out" \
        --error="${LOG_DIR}/${job_name}.err" \
        validation/sweeping/run_job_validate_2fold.sh \
          --model_name "${model_name}" \
          --num_heads "${num_heads}" \
          --alpha "${alpha}" \
          --instruction_prompt "${INSTRUCTION_PROMPT}" \
          --judge_name "${JUDGE_NAME}" \
          --info_name "${INFO_NAME}" \
          --seed "${seed}"
      sleep 60
    done
  done
done
