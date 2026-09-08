"""Download the distilled DistilBERT router checkpoint from the Hugging Face Hub.

Usage:
    python -m atr.tools.download_router --out_dir ./models/atr_router

The checkpoint is what `atr.online.main --router_type learned --router_model_path`
expects. It is trained with `atr.tools.train_router train` on
`data/oracle_labels_distill_mixed.jsonl` (mixed-dataset oracle labels).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO = os.getenv("ATR_ROUTER_REPO", "<ANONYMIZED>/atr-router-distilbert")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the ATR distilled router from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id", default=DEFAULT_REPO,
        help=f"HF Hub repo id (default: {DEFAULT_REPO}; override with "
             f"ATR_ROUTER_REPO env var).",
    )
    parser.add_argument(
        "--revision", default="main",
        help="Branch / tag / commit on the HF repo.",
    )
    parser.add_argument(
        "--out_dir", default="./models/atr_router",
        help="Where to download the checkpoint (default: ./models/atr_router).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    out = Path(args.out_dir).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id}@{args.revision} → {out}")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(out),
    )
    print(f"Done. Pass --router_model_path {out} to atr.online.main.")


if __name__ == "__main__":
    main()
