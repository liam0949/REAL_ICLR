#!/usr/bin/env python3
"""
Convert learned REAL head rankings into LoFiT head format.

Input:
- REAL/OneForAll results in `results_dump/head_sorted`
  loaded via `utils.get_vq_top_heads(...)`.

Output:
- `.npy` file with shape [N, 2], each row is `[layer, head]`,
  sorted by learned importance (high -> low), for LoFiT consumption.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hl_paths import resolve_dir
from utils import get_vq_top_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export learned REAL heads to LoFiT .npy format ([layer, head])."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model key used in OneForAll outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used in OneForAll outputs.")
    parser.add_argument("--num_fold", type=int, default=2, help="Total fold count when --fold is not set.")
    parser.add_argument("--fold", type=int, default=None, help="Export one fold only (default: export all folds).")
    parser.add_argument("--num_heads", type=int, default=0, help="Top-K heads to export; <=0 means all non-zero heads.")
    parser.add_argument("--codebook", type=int, default=32, help="Codebook used in OneForAll filename; release default is 32.")
    parser.add_argument("--units", type=int, default=8, help="Units used in OneForAll filename; release default is 8.")
    parser.add_argument("--per_ratio", type=float, default=0.0, help="Perplexity ratio used in OneForAll filename.")
    parser.add_argument("--min_score", type=float, default=None, help="Optional score threshold after ranking.")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results_dump root.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: results_dump/head_sorted_lof).")
    parser.add_argument("--tag", type=str, default="", help="Optional suffix in output filename.")
    return parser.parse_args()


def _folds_to_export(args: argparse.Namespace) -> List[int]:
    if args.fold is not None:
        return [args.fold]
    return list(range(args.num_fold))


def main() -> None:
    args = parse_args()

    results_root = resolve_dir(args.results_dir, "HONEST_LLAMA_RESULTS_DIR", "validation/results_dump")
    os.environ["HONEST_LLAMA_RESULTS_DIR"] = str(results_root)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_root / "head_sorted_lof"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_head = args.num_heads <= 0
    k = 1 if all_head else args.num_heads

    for fold in _folds_to_export(args):
        ranked = get_vq_top_heads(
            model_name_t=args.model_name,
            k=k,
            i=fold,
            seed=args.seed,
            codebook=args.codebook,
            units=args.units,
            per_ratio=args.per_ratio,
            all_head=all_head,
        )

        if args.min_score is not None:
            ranked = [item for item in ranked if float(item[1]) >= args.min_score]

        lofit_heads = np.asarray(
            [[int(layer), int(head)] for (layer, head), _score in ranked],
            dtype=np.int64,
        )

        top_tag = "all" if all_head else str(args.num_heads)
        out_name = f"{args.model_name}_seed_{args.seed}_REAL_top_{top_tag}_heads_fold_{fold}"
        if args.tag:
            out_name += f"_{args.tag}"
        out_path = output_dir / f"{out_name}.npy"

        np.save(out_path, lofit_heads)
        print(f"[OK] fold={fold} heads={lofit_heads.shape[0]} -> {out_path}")


if __name__ == "__main__":
    main()
