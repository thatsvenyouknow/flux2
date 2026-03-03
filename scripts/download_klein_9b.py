#!/usr/bin/env python3
"""
Download FLUX.2 Klein 9B model (and shared AE) from Hugging Face to a local directory,
then print the environment variables to use with the pipeline.

The 9B repo contains: flow weights, autoencoder, text_encoder, and tokenizer.
After running this script, set the printed env vars (or source the suggested file)
so that Flux2Pipeline and the retouching pipeline use the local copy.

Usage:
    python scripts/download_klein_9b.py
    python scripts/download_klein_9b.py --dir /home/ubuntu/data/models/FLUX.2-klein-9B
    python scripts/download_klein_9b.py --no-ae

Requires: pip install huggingface_hub
Login for gated models: huggingface-cli login
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "black-forest-labs/FLUX.2-klein-9B"
REPO_ID_FP8 = "black-forest-labs/FLUX.2-klein-9b-fp8"
FLOW_FILENAME = "flux-2-klein-9b.safetensors"
FP8_FILENAME = "flux-2-klein-9b-fp8.safetensors"
AE_FILENAME = "ae.safetensors"
DEFAULT_DOWNLOAD_DIR = Path("/home/ubuntu/data/models/FLUX.2-klein-9B")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FLUX.2 Klein 9B model and print env vars for the pipeline.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_DOWNLOAD_DIR),
        help=f"Directory to download into (default: {DEFAULT_DOWNLOAD_DIR}).",
    )
    parser.add_argument(
        "--no-ae",
        action="store_true",
        help="Skip downloading AE (use if you already have ae.safetensors from 4B).",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        default=True,
        help="Also download the FP8 checkpoint (~9.4GB, half the bf16 size). On by default.",
    )
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="Skip downloading the FP8 checkpoint.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print env vars for an existing directory (no download).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        sys.exit("Install huggingface_hub: pip install huggingface_hub")

    download_fp8 = args.fp8 and not args.no_fp8

    if args.print_only:
        if not args.dir or not Path(args.dir).exists():
            sys.exit("--print-only requires --dir pointing to an existing repo directory.")
        repo_root = Path(args.dir).resolve()
    else:
        local_dir = Path(args.dir).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {REPO_ID} to {local_dir}...")
        repo_root = Path(
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(local_dir),
                repo_type="model",
            )
        )

        print(f"Downloaded to: {repo_root}")

        if not args.no_ae:
            flow_path = repo_root / FLOW_FILENAME
            ae_path = repo_root / AE_FILENAME
            if not flow_path.exists():
                print(f"Warning: {flow_path} not found after download.", file=sys.stderr)
            if not ae_path.exists():
                print(f"Warning: {ae_path} not found; AE may be in 4B repo.", file=sys.stderr)

        if download_fp8:
            fp8_dest = repo_root / FP8_FILENAME
            if fp8_dest.exists():
                print(f"FP8 checkpoint already exists: {fp8_dest}")
            else:
                import shutil
                print(f"Downloading FP8 checkpoint from {REPO_ID_FP8}...")
                cached = hf_hub_download(
                    repo_id=REPO_ID_FP8,
                    filename=FP8_FILENAME,
                    repo_type="model",
                )
                shutil.copy2(cached, str(fp8_dest))
                print(f"  Saved FP8 checkpoint: {fp8_dest}")

    flow_path = repo_root / FLOW_FILENAME
    fp8_path = repo_root / FP8_FILENAME
    ae_path = repo_root / AE_FILENAME

    exports = [
        f"export KLEIN_9B_MODEL_PATH=\"{flow_path}\"",
        f"export FLUX2_KLEIN_9B_REPO=\"{repo_root}\"",
    ]
    if fp8_path.exists():
        exports.append(f"export KLEIN_9B_FP8_MODEL_PATH=\"{fp8_path}\"")
    if ae_path.exists():
        exports.append(f"export AE_MODEL_PATH=\"{ae_path}\"")

    print("\n" + "=" * 60)
    print("Set these environment variables to use the Klein 9B model:")
    print("=" * 60)
    for line in exports:
        print(line)
    print("=" * 60)
    print("\nOr run: eval \"$(python scripts/download_klein_9b.py --print-only --dir " + str(repo_root) + ")\"")
    print("\nThen run the pipeline with the FP8 model (recommended, ~9.4 GB VRAM for flow):")
    print("  PYTHONPATH=src python scripts/retouching.py --image <path> --flux-model flux.2-klein-9b-fp8")


if __name__ == "__main__":
    main()
