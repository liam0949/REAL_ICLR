# REAL

Official code repository for:

**REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering**  
Li-Ming Zhan, Bo LIU, Yujie Feng, Chengqiang Xie, Jiannong Cao, Xiao-Ming Wu  
ICLR 2026 Poster

Paper: https://openreview.net/forum?id=P38RYdkFLI

This release reproduces the TruthfulQA experiments. The codebase builds on
[`likenneth/honest_llama`](https://github.com/likenneth/honest_llama) and adapts
its feature extraction, TruthfulQA evaluation, and pyvene-based intervention code.
The REAL training entry point is `validation/OneForAll.py`; the TruthfulQA
evaluation entry point is `validation/validate_2fold.py`.

## Reproduction Overview

The main reproduction path is:

1. Build and enter the Docker environment.
2. Prepare access to the target Hugging Face model or a local checkpoint.
3. Extract TruthfulQA activation features into `features/`.
4. Train REAL head rankings with `validation/OneForAll.py`.
5. Evaluate TruthfulQA with `validation/validate_2fold.py --method REAL`.

The repository includes:

- TruthfulQA package/data under `TruthfulQA/`.
- Seed-42 two-fold splits under `validation/splits/`.
- Camera-ready REAL head rankings under `validation/results_dump/head_sorted/`.
- An empty `features/` directory placeholder.

Large activation banks, answer dumps, summary dumps, logs, and non-TruthfulQA
intermediate results are intentionally excluded.

## 1. Build The Environment

This release is intended to run inside Docker. The runtime dependencies are listed
in `requirements.txt`.

Host prerequisites:

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit

Build the image from the repository root:

```bash
docker build -t real-release:latest .
```

Start an interactive container. `REAL_MODEL_DIR` is mounted to `/models`; point it
to either a Hugging Face cache directory or a directory containing local checkpoints.

```bash
export REAL_MODEL_DIR=/path/to/your/model_or_hf_cache_dir
./docker.sh
```

Inside the container:

```bash
cd /workspace/app
python - <<'PY'
import torch, transformers, pyvene
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda available", torch.cuda.is_available())
PY
```

The Dockerfile defaults to `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel`. To use a
cluster-specific compatible PyTorch/CUDA image:

```bash
docker build \
  --build-arg BASE_IMAGE=YOUR_PYTORCH_CUDA_DEVEL_IMAGE \
  -t real-release:latest .
```

For gated Hugging Face models, run `huggingface-cli login` inside the container or
export `HF_TOKEN` on the host before `./docker.sh`.

## 2. Prepare Model Resolution

Use the same `{your_model_name}` across feature extraction, REAL training, and
validation. Bundled REAL rankings are provided for:

```text
llama2_7B
llama2-chat-7B
Llama-2-13b-chat-hf
Qwen2.5-7B
Qwen2.5-7B-Instruct
```

By default, `hl_config.py` resolves these names to public Hugging Face model ids.
For local checkpoints mounted under `/models`, either pass `--model_path` in each
command or create a JSON registry:

```json
{
  "llama2_7B": "/models/Llama-2-7b-hf",
  "llama2-chat-7B": "/models/Llama-2-7b-chat-hf",
  "Llama-2-13b-chat-hf": "/models/Llama-2-13b-chat-hf",
  "Qwen2.5-7B": "/models/Qwen2.5-7B",
  "Qwen2.5-7B-Instruct": "/models/Qwen2.5-7B-Instruct"
}
```

Then export it inside the container:

```bash
export HONEST_LLAMA_MODEL_REGISTRY=/workspace/app/model_registry.json
```

## 3. Extract TruthfulQA Features

Run from `/workspace/app` inside the container:

```bash
CUDA_VISIBLE_DEVICES=0 bash get_activations/get_activations.sh \
  --model_name {your_model_name} \
  --device 0
```

This wrapper extracts the two feature banks used by REAL:

```bash
CUDA_VISIBLE_DEVICES=0 python get_activations/get_activations.py \
  --model_name {your_model_name} \
  --dataset_name tqa_mc2 \
  --device 0 \
  --require_hf

CUDA_VISIBLE_DEVICES=0 python get_activations/get_activations.py \
  --model_name {your_model_name} \
  --dataset_name tqa_gen_end_q \
  --device 0 \
  --require_hf
```

Expected outputs:

```text
features/{your_model_name}_tqa_mc2_head_wise.npy
features/{your_model_name}_tqa_mc2_labels.npy
features/{your_model_name}_tqa_gen_end_q_head_wise.npy
features/{your_model_name}_tqa_gen_end_q_labels.npy
```

The default attention backend is `flash_attention_2`. If it is unavailable, the
scripts fall back to the Transformers default backend. You can also explicitly add
`--attn_implementation eager`.

## 4. Train REAL Head Rankings

This release uses `codebook=32` and `units=8`.

Run from `/workspace/app/validation`:

```bash
cd validation
CUDA_VISIBLE_DEVICES=0 python OneForAll.py \
  --model_name {your_model_name} \
  --device 0 \
  --seed 42 \
  --codebook 32 \
  --units 8 \
  --c_loss_scale 0.001 \
  --epochs_vq 20 \
  --epochs_gru 10 \
  --batch_size 32
```

The trained ranking is written to:

```text
validation/results_dump/head_sorted/
```

If you only want to validate the camera-ready rankings included in this release,
you may skip this step for the bundled model names.

## 5. Evaluate REAL On TruthfulQA

Run from `/workspace/app/validation`:

```bash
cd validation
CUDA_VISIBLE_DEVICES=0 python validate_2fold.py \
  --model_name {your_model_name} \
  --method REAL \
  --num_heads 48 \
  --alpha 15 \
  --codebook 32 \
  --units 8 \
  --per_ratio 0.0 \
  --use_center_of_mass \
  --device 0
```

Validation writes:

```text
validation/results_dump/answer_dump/
validation/results_dump/summary_dump/
```

If your cluster uses a Hugging Face mirror, set `HF_ENDPOINT` before running:

```bash
export HF_ENDPOINT=YOUR_HF_ENDPOINT
```

## Optional ITI Baseline

`REAL` is our method. The deprecated alias `vq` is still accepted for backward
compatibility. `iti` is the logistic-probe ITI baseline.

To run the ITI baseline with the same validation entry point:

```bash
cd validation
CUDA_VISIBLE_DEVICES=0 python validate_2fold.py \
  --model_name {your_model_name} \
  --method iti \
  --num_heads 48 \
  --alpha 15 \
  --use_center_of_mass \
  --device 0
```

## Useful Overrides

Path and model overrides:

- `--model_path`
- `--model_registry`
- `--features_dir`
- `--results_dir`
- `--splits_dir`
- `--truthfulqa_dir`
- `--cache_dir`
- `HONEST_LLAMA_MODEL_REGISTRY`
- `HONEST_LLAMA_CACHE_DIR`

If you use GPT-based TruthfulQA judging, pass `OPENAI_API_KEY` into the container
and provide judge/info model names through `--judge_name` and `--info_name`.

## Citation

If you use this repository, please cite REAL:

```bibtex
@inproceedings{
zhan2026real,
title={{REAL}: Reading Out Transformer Activations for Precise Localization in Language Model Steering},
author={Li-Ming Zhan and Bo LIU and Yujie Feng and Chengqiang Xie and Jiannong Cao and Xiao-Ming Wu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=P38RYdkFLI}
}
```

This codebase is based on Honest LLaMA and adapts its feature extraction and
validation pipeline. Please also cite Honest LLaMA / ITI where appropriate:

```bibtex
@article{li2024inference,
  title={Inference-time intervention: Eliciting truthful answers from a language model},
  author={Li, Kenneth and Patel, Oam and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## License

This repository is released under the MIT License, preserving the original Honest
LLaMA license notice. The bundled TruthfulQA package/data is under Apache License
2.0. See `NOTICE.md` for details.
