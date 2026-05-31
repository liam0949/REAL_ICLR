from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent

# Central model registry. Override via HONEST_LLAMA_MODEL_REGISTRY or --model_registry.
# Values are public Hugging Face model ids by default. Use --model_path for a
# local checkpoint directory.
DEFAULT_MODEL_REGISTRY: Dict[str, str] = {
    "llama2_7B": "meta-llama/Llama-2-7b-hf",
    "llama2-chat-7B": "meta-llama/Llama-2-7b-chat-hf",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
}


def _load_registry_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Model registry must be a JSON object, got {type(data)}")
    return {str(k): str(v) for k, v in data.items()}


def load_model_registry(registry_path: Optional[str] = None) -> Dict[str, str]:
    registry = dict(DEFAULT_MODEL_REGISTRY)
    env_path = os.getenv("HONEST_LLAMA_MODEL_REGISTRY")
    for path in (registry_path, env_path):
        if path:
            registry.update(_load_registry_from_json(path))
    return registry


def resolve_model_path(
    model_name: str,
    model_prefix: str = "",
    *,
    model_path: Optional[str] = None,
    registry_path: Optional[str] = None,
    registry: Optional[Dict[str, str]] = None,
) -> str:
    if model_path:
        return model_path
    reg = registry or load_model_registry(registry_path)
    key = f"{model_prefix}{model_name}"
    if key in reg:
        return reg[key]
    if model_name in reg:
        return reg[model_name]
    raise KeyError(
        f"Model '{key}' not found in registry. "
        "Provide --model_path or --model_registry (JSON), "
        "or set HONEST_LLAMA_MODEL_REGISTRY."
    )


def resolve_cache_dir(cache_dir: Optional[str] = None) -> Optional[str]:
    return cache_dir or os.getenv("HONEST_LLAMA_CACHE_DIR")
