#!/usr/bin/env bash
set -euo pipefail

# Sequential validation sweep template.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/validation"

MODEL_NAME="${MODEL_NAME:-llama2_7B}"
MODEL_PREFIX="${MODEL_PREFIX:-}"
INSTRUCTION_PROMPT="${INSTRUCTION_PROMPT:-default}"
JUDGE_NAME="${JUDGE_NAME:-<JUDGE_MODEL_NAME>}"
INFO_NAME="${INFO_NAME:-<INFO_MODEL_NAME>}"
DEVICE="${DEVICE:-0}"

for alpha in 15; do
  for num_heads in 48; do
    for seed in 42; do
      echo "alpha=${alpha} num_heads=${num_heads} seed=${seed}"
      cmd=(
        python validate_2fold.py
        --model_name "${MODEL_NAME}"
        --method REAL
        --num_heads "${num_heads}"
        --alpha "${alpha}"
        --codebook 32
        --units 8
        --per_ratio 0.0
        --instruction_prompt "${INSTRUCTION_PROMPT}"
        --device "${DEVICE}"
        --num_fold 2
        --use_center_of_mass
        --judge_name "${JUDGE_NAME}"
        --info_name "${INFO_NAME}"
        --seed "${seed}"
      )
      if [[ -n "${MODEL_PREFIX}" ]]; then
        cmd+=(--model_prefix "${MODEL_PREFIX}")
      fi
      "${cmd[@]}"
    done
  done
done
