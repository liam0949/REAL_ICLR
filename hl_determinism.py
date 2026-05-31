from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_determinism(seed: int, *, strict: bool = True) -> None:
    """
    Set seeds and deterministic flags for Python, NumPy, and PyTorch.
    If strict is True, disables TF32 and cuDNN benchmark for repeatability.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = not strict
    torch.backends.cudnn.allow_tf32 = not strict

    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    except Exception:
        # Older PyTorch versions may not support this
        pass
