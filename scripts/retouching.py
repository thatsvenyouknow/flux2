#!/usr/bin/env python3
"""
CLI for the TUI local retouching pipeline.

Usage:
    PYTHONPATH=src python scripts/retouching.py --image path/to/hotel.jpg
    PYTHONPATH=src python scripts/retouching.py --image path/to/hotel.jpg --two-pass
    PYTHONPATH=src python scripts/retouching.py --image path/to/hotel.jpg --skip-vlm --prompts "logo,smoke detector"
    PYTHONPATH=src python scripts/retouching.py --image path/to/hotel.jpg --prompt "Remove the fire extinguisher."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def pick_image(image_arg: str | None) -> Path:
    if image_arg:
        p = Path(image_arg)
        if not p.exists():
            sys.exit(f"Image not found: {p}")
        return p
    sys.exit("--image is required. Pass a path to the input image.")


def main():
    parser = argparse.ArgumentParser(
        description="TUI hotel/resort image retouching via Gemini + SAM 3 + Flux2 inpainting.",
    )

    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory for results (default: output/).")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save debug mask images alongside the output.")

    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip Gemini VLM analysis; use default SAM prompts.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override edit prompt (skip VLM prompt generation).")
    parser.add_argument("--task", type=str, default=None,
                        help="Extra instruction for the VLM on top of the config.")
    parser.add_argument("--vlm-model", type=str, default="gemini-2.5-pro",
                        help="Gemini model (gemini-2.5-pro, gemini-2.0-pro, gemini-2.0-flash).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to retouching YAML config (uses bundled default).")

    parser.add_argument("--prompts", type=str, default=None,
                        help="Comma-separated SAM prompts (overrides VLM + defaults). "
                             "Applied as remove prompts.")
    parser.add_argument("--sam-model", type=str, default="facebook/sam3",
                        help="SAM 3 Hugging Face model id.")
    parser.add_argument("--sam-remove-threshold", type=float, default=0.35,
                        help="SAM confidence for objects/logos (default: 0.35, lower = more sensitive).")
    parser.add_argument("--sam-retouch-threshold", type=float, default=0.65,
                        help="SAM confidence for surfaces/stains (default: 0.65, higher = more certain).")
    parser.add_argument("--dilation", type=int, default=8,
                        help="Mask dilation in pixels (default: 8).")
    parser.add_argument("--max-side", type=int, default=2048,
                        help="Max image dimension — large images are downscaled at the start (default: 2048).")

    parser.add_argument("--two-pass", action="store_true", default=True,
                        help="Two-pass mode: remove then retouch (default: on).")
    parser.add_argument("--single-pass", action="store_true",
                        help="Single-pass mode with combined mask.")
    parser.add_argument("--remove-strength", type=float, default=0.85,
                        help="Inpainting strength for remove pass.")
    parser.add_argument("--retouch-strength", type=float, default=0.55,
                        help="Inpainting strength for retouch pass.")
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Flux2 denoising steps.")
    parser.add_argument("--guidance", type=float, default=4.0,
                        help="Flux2 guidance scale.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")

    parser.add_argument("--flux-model", type=str, default="flux.2-klein-9b-fp8",
                        help="Flux2 model name (default: flux.2-klein-9b-fp8).")
    parser.add_argument("--cpu-offload", action="store_true", default=False,
                        help="CPU offloading: text encoder → CPU after encoding (off by default; "
                             "enable for GPUs < 40 GiB or higher --max-side values).")
    parser.add_argument("--no-cpu-offload", action="store_true",
                        help="Explicitly disable CPU offloading (all models stay on GPU).")
    parser.add_argument("--no-quantize", action="store_true", default=False,
                        help="Disable INT8 text encoder quantization (INT8 is incompatible with --cpu-offload).")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for Flux2.")

    args = parser.parse_args()

    image_path = pick_image(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    two_pass = not args.single_pass

    remove_prompts = None
    retouch_prompts = None
    if args.prompts:
        remove_prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        retouch_prompts = []

    cpu_offload = args.cpu_offload and not args.no_cpu_offload
    quantize_te = not args.no_quantize and not cpu_offload  # INT8 can't be moved to CPU

    print(f"Loading Flux2 ({args.flux_model}, cpu_offload={cpu_offload})...")
    from flux2.pipeline import Flux2Pipeline
    flux = Flux2Pipeline(
        model_name=args.flux_model,
        cpu_offloading=cpu_offload,
        quantize_text_encoder=quantize_te,
        compile_model=args.compile,
    )

    from bertelsman.pipeline import TUIRetouchingPipeline, save_run_log
    pipeline = TUIRetouchingPipeline(
        flux_pipeline=flux,
        config_path=args.config,
        sam_model_id=args.sam_model,
        vlm_model=args.vlm_model,
    )

    masks_dir = output_dir / "masks" if args.save_masks else None
    result = pipeline.run(
        image_path,
        extra_task=args.task,
        skip_vlm=args.skip_vlm,
        manual_prompt=args.prompt,
        remove_prompts=remove_prompts,
        retouch_prompts=retouch_prompts,
        dilation_px=args.dilation,
        sam_remove_threshold=args.sam_remove_threshold,
        sam_retouch_threshold=args.sam_retouch_threshold,
        two_pass=two_pass,
        remove_strength=args.remove_strength,
        retouch_strength=args.retouch_strength,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        max_side=args.max_side,
        save_masks_dir=masks_dir,
    )

    stem = image_path.stem
    out_path = output_dir / f"{stem}_retouched.png"
    result.image.save(out_path, quality=95, subsampling=0)
    print(f"\nSaved: {out_path}")

    log_path = save_run_log(image_path, out_path, result)
    if log_path:
        print(f"Log:   {log_path}")


if __name__ == "__main__":
    main()
