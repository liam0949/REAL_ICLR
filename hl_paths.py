from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent


def resolve_dir(cli_value: Optional[str], env_var: str, default_relative: str) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (REPO_ROOT / default_relative).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
