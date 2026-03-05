#!/usr/bin/env python3
"""
Visual evaluation of the TUI retouching pipeline.

Two test modes:

  masks   — Evaluate SAM + VLM mask quality.
             Output figure: N rows x 3 columns [Original | Mask overlay | Target]

  pipeline — Evaluate the full inpainting pipeline.
             Output figure: N rows x 3 columns [Original | Edited | Target]

Usage:
    # SAM + VLM mask accuracy (default prompts, no VLM)
    PYTHONPATH=src python -m bertelsman.tests.test_retouching masks

    # SAM + VLM mask accuracy (VLM-driven prompts)
    PYTHONPATH=src python -m bertelsman.tests.test_retouching masks --use-vlm

    # Full pipeline
    PYTHONPATH=src python -m bertelsman.tests.test_retouching pipeline

    # Full pipeline, single-pass, 3 images
    PYTHONPATH=src python -m bertelsman.tests.test_retouching pipeline --n 3 --single-pass

    # Both tests in sequence
    PYTHONPATH=src python -m bertelsman.tests.test_retouching all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

BERTELSMAN_DATA = Path("/home/ubuntu/bertelsman/data")
INPUT_DIR = BERTELSMAN_DATA / "input_images"
TARGET_DIR = BERTELSMAN_DATA / "target_output"

DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/flux2/output/tests")


def find_paired_images(
    input_dir: Path, target_dir: Path, n: int,
) -> list[dict]:
    """Find input images that have a matching target, up to n pairs."""
    inputs = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    pairs = []
    for inp in inputs:
        if len(pairs) >= n:
            break
        target = _find_target(inp, target_dir)
        pairs.append({"input": inp, "target": target, "name": inp.stem})
    return pairs


def _find_target(input_path: Path, target_dir: Path) -> Optional[Path]:
    stem = input_path.stem
    for ext in [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp"]:
        candidate = target_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Overlay helper (self-contained so we don't depend on the old bertelsman)
# ---------------------------------------------------------------------------

def create_mask_overlay(
    image: Image.Image,
    combined_mask: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Red-tinted overlay of the mask on the original image."""
    img_arr = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img_arr.shape[:2]
    if combined_mask.shape[0] != h or combined_mask.shape[1] != w:
        combined_mask = np.array(
            Image.fromarray(combined_mask).resize((w, h), Image.NEAREST)
        )
    mask_norm = (combined_mask > 0).astype(np.float32)[:, :, np.newaxis]
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    overlay = img_arr * (1 - alpha * mask_norm) + red * mask_norm * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def _format_sam_labels(sam_labels: list[str], max_lines: int = 12) -> str:
    """Format SAM labels as a compact multi-line string for figure annotations."""
    if not sam_labels:
        return "(no prompts)"
    lines = [f"- {lbl}" for lbl in sam_labels[:max_lines]]
    if len(sam_labels) > max_lines:
        lines.append(f"  (+{len(sam_labels) - max_lines} more)")
    return "\n".join(lines)


def build_figure(
    rows: list[dict],
    col_keys: list[str],
    col_titles: list[str],
    output_path: Path,
    suptitle: str = "",
) -> None:
    """Generic N-row x M-col comparison figure.

    Each row dict may contain a ``"sam_labels"`` key (list[str]) which will be
    rendered in a narrow text column on the left.  If absent, nothing is shown.
    """
    n = len(rows)
    ncols = len(col_keys)
    has_labels = any(row.get("sam_labels") for row in rows)

    if has_labels:
        label_width = 2.0
        img_width = 7
        widths = [label_width] + [img_width] * ncols
        fig, all_axes = plt.subplots(
            n, ncols + 1,
            figsize=(sum(widths), 6 * n),
            gridspec_kw={"width_ratios": widths},
        )
        if n == 1:
            all_axes = [all_axes]
        label_axes = [row_axes[0] for row_axes in all_axes]
        img_axes = [row_axes[1:] for row_axes in all_axes]
    else:
        fig, img_axes = plt.subplots(n, ncols, figsize=(7 * ncols, 6 * n))
        if n == 1:
            img_axes = [img_axes]
        label_axes = [None] * n

    for col_idx, title in enumerate(col_titles):
        img_axes[0][col_idx].set_title(title, fontsize=16, fontweight="bold", pad=12)

    if has_labels and label_axes[0] is not None:
        label_axes[0].set_title("SAM prompts", fontsize=11, fontweight="bold", pad=12)

    for row_idx, row in enumerate(rows):
        lax = label_axes[row_idx]
        if lax is not None:
            lax.axis("off")
            sam_labels = row.get("sam_labels", [])
            txt = _format_sam_labels(sam_labels)
            lax.text(
                0.05, 0.5, txt,
                transform=lax.transAxes,
                fontsize=6, fontfamily="monospace",
                va="center", ha="left",
                linespacing=1.4,
            )

        for col_idx, key in enumerate(col_keys):
            ax = img_axes[row_idx][col_idx]
            img = row.get(key)
            if img is not None:
                if isinstance(img, Path):
                    img = Image.open(img).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(
                    0.5, 0.5, f"No {col_titles[col_idx].lower()}",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.01)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"\nFigure saved: {output_path}")
    except PermissionError:
        fallback = Path("/tmp") / output_path.name
        fig.savefig(fallback, dpi=120, bbox_inches="tight")
        print(f"\n(permission denied) Figure saved: {fallback}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 1: Mask accuracy
# ---------------------------------------------------------------------------

def test_masks(
    pairs: list[dict],
    output_path: Path,
    *,
    use_vlm: bool = False,
    vlm_model: str = "gemini-3.1-pro-preview",
    sam_model: str = "facebook/sam3",
    sam_remove_threshold: float = 0.40,
    sam_retouch_threshold: float = 0.65,
    dilation_px: int = 8,
    mode: str = "remove",
    overlay_alpha: float = 0.45,
) -> None:
    """Test 1: evaluate SAM mask quality against target images."""
    from bertelsman.masking import (
        PROMPTS_REMOVE,
        PROMPTS_RETOUCH,
        SAM3Session,
        generate_masks,
    )

    if use_vlm:
        from bertelsman.vlm import analyze_image, load_config
        config = load_config()

    sam_session = SAM3Session(model_id=sam_model)

    rows = []
    for pair in pairs:
        img_path = pair["input"]
        print(f"  SAM: {img_path.name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            remove_prompts = None
            retouch_prompts = None
            sam_labels: list[str] = []

            if use_vlm:
                analysis = analyze_image(img_path, config=config, model=vlm_model)
                remove_prompts = analysis.remove_sam_prompts
                retouch_prompts = analysis.retouch_sam_prompts
                print(f"VLM found {len(analysis.issues)} issues "
                      f"({len(analysis.remove_issues)} remove, "
                      f"{len(analysis.retouch_issues)} retouch)...",
                      end=" ", flush=True)

            if mode == "remove":
                if remove_prompts is None:
                    remove_prompts = list(PROMPTS_REMOVE)
                retouch_prompts = []
            elif mode == "retouch":
                if retouch_prompts is None:
                    retouch_prompts = list(PROMPTS_RETOUCH)
                remove_prompts = []

            sam_labels = (remove_prompts or []) + (retouch_prompts or [])

            mask_result = generate_masks(
                img_path,
                remove_prompts=remove_prompts,
                retouch_prompts=retouch_prompts,
                model_id=sam_model,
                remove_threshold=sam_remove_threshold,
                retouch_threshold=sam_retouch_threshold,
                dilation_px=dilation_px,
                session=sam_session,
            )

            original = Image.open(img_path).convert("RGB")
            if mode == "remove":
                mask_for_overlay = mask_result.remove_combined
            elif mode == "retouch":
                mask_for_overlay = mask_result.retouch_combined
            else:
                mask_for_overlay = mask_result.all_combined
            overlay = create_mask_overlay(original, mask_for_overlay, alpha=overlay_alpha)

            n_masks = len(mask_result.remove_masks) + len(mask_result.retouch_masks)
            elapsed = time.time() - t0
            print(f"ok ({n_masks} masks, {elapsed:.1f}s)")

            rows.append({
                "original": original,
                "overlay": overlay,
                "target": pair["target"],
                "name": pair["name"],
                "sam_labels": sam_labels,
            })

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    if not rows:
        print("All images failed. Nothing to plot.")
        return

    vlm_tag = " (VLM-driven)" if use_vlm else " (default prompts)"
    build_figure(
        rows,
        col_keys=["original", "overlay", "target"],
        col_titles=["Original", "SAM Mask" + vlm_tag, "Target"],
        output_path=output_path,
        suptitle=f"SAM Mask Accuracy — {len(rows)} images, mode={mode}",
    )


# ---------------------------------------------------------------------------
# Test 2: Full pipeline
# ---------------------------------------------------------------------------

def test_pipeline(
    pairs: list[dict],
    output_path: Path,
    *,
    flux_model: str = "flux.2-klein-9b-fp8",
    fast: bool = False,
    vlm_model: str = "gemini-3.1-pro-preview",
    sam_model: str = "facebook/sam3",
    skip_vlm: bool = False,
    two_pass: bool = False,
    remove_strength: float = 1.0,
    retouch_strength: float = 0.55,
    num_steps: int = 8,
    guidance: float = 4.0,
    seed: int | None = None,
    sam_remove_threshold: float = 0.4,
    sam_retouch_threshold: float = 0.65,
    dilation_px: int = 8,
    max_side: int = 2048,
) -> None:
    """Test 2: run the full retouching pipeline and compare against targets."""
    from flux2.pipeline import Flux2Pipeline
    from bertelsman.pipeline import TUIRetouchingPipeline

    if fast:
        flux_model = "flux.2-klein-4b"  # 4B loads much faster than 9B (fewer weights)

    # With early downscaling (max_side=2048) peak VRAM is ~31 GiB on a 40 GiB GPU,
    # so all models fit on GPU without offloading.  Re-enable cpu_offloading for
    # smaller GPUs or higher max_side values.
    print(f"Loading Flux2 ({flux_model}, cpu_offloading=False)...")
    flux = Flux2Pipeline(model_name=flux_model, quantize_text_encoder=False, cpu_offloading=False)

    pipeline = TUIRetouchingPipeline(
        flux_pipeline=flux,
        sam_model_id=sam_model,
        vlm_model=vlm_model,
    )

    rows = []
    for pair in pairs:
        img_path = pair["input"]
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*60}")
        t0 = time.time()

        try:
            result = pipeline.run(
                img_path,
                skip_vlm=skip_vlm,
                two_pass=two_pass,
                remove_strength=remove_strength,
                retouch_strength=retouch_strength,
                num_steps=num_steps,
                guidance=guidance,
                seed=seed,
                sam_remove_threshold=sam_remove_threshold,
                sam_retouch_threshold=sam_retouch_threshold,
                dilation_px=dilation_px,
                max_side=max_side,
            )

            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s (passes: {result.passes})")

            # The pipeline may downscale the input — use the edited image's
            # size as the working resolution so the mask overlay lines up.
            edit_w, edit_h = result.image.size
            original = Image.open(img_path).convert("RGB")
            if original.size != (edit_w, edit_h):
                original = original.resize((edit_w, edit_h), Image.LANCZOS)
            if result.mask_result is not None and result.mask_result.all_combined is not None:
                masked = create_mask_overlay(
                    original,
                    result.mask_result.all_combined,
                    alpha=0.45,
                )
            else:
                masked = original

            sam_labels: list[str] = []
            if result.analysis:
                sam_labels = result.analysis.remove_sam_prompts

            rows.append({
                "original": original,
                "masked": masked,
                "edited": result.image,
                "target": pair["target"],
                "name": pair["name"],
                "sam_labels": sam_labels,
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not rows:
        print("All images failed. Nothing to plot.")
        return

    mode_tag = "two-pass" if two_pass else "single-pass"
    vlm_tag = "no VLM" if skip_vlm else "VLM"
    build_figure(
        rows,
        col_keys=["original", "masked", "edited", "target"],
        col_titles=["Original", "Original + Mask", "Edited", "Target"],
        output_path=output_path,
        suptitle=f"Pipeline Evaluation — {len(rows)} images, {mode_tag}, {vlm_tag}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual evaluation of the TUI retouching pipeline.",
    )
    parser.add_argument(
        "test", choices=["masks", "pipeline", "all"],
        help="Which test to run: 'masks' (SAM accuracy), 'pipeline' (full), or 'all'.",
    )
    parser.add_argument("--n", type=int, default=10,
                        help="Number of images to evaluate (default: 10).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for output figures.")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR),
                        help="Directory with input images.")
    parser.add_argument("--target-dir", type=str, default=str(TARGET_DIR),
                        help="Directory with target/reference images.")

    # Mask test options
    parser.add_argument("--use-vlm", action="store_true",
                        help="[masks] Use VLM to derive SAM prompts (otherwise use defaults).")
    parser.add_argument("--mode", choices=["all", "remove", "retouch"], default="remove",
                        help="[masks] SAM concept category (default: remove).")
    parser.add_argument("--overlay-alpha", type=float, default=0.45,
                        help="[masks] Mask overlay opacity (default: 0.45).")

    # Pipeline test options
    parser.add_argument("--skip-vlm", action="store_true",
                        help="[pipeline] Skip VLM, use default SAM prompts + generic prompt.")
    parser.add_argument("--single-pass", action="store_true",
                        help="[pipeline] Single inpainting pass instead of two-pass.")
    parser.add_argument("--remove-strength", type=float, default=1.0,
                        help="[pipeline] Inpainting strength for remove pass (1.0 = full removal).")
    parser.add_argument("--retouch-strength", type=float, default=0.55,
                        help="[pipeline] Inpainting strength for retouch pass.")
    parser.add_argument("--num-steps", type=int, default=8,
                        help="[pipeline] Flux2 denoising steps (more = better fill quality).")
    parser.add_argument("--guidance", type=float, default=4.0,
                        help="[pipeline] Flux2 guidance scale.")
    parser.add_argument("--seed", type=int, default=None,
                        help="[pipeline] Random seed.")
    parser.add_argument("--flux-model", type=str, default="flux.2-klein-9b-fp8",
                        help="[pipeline] Flux2 model (flux.2-klein-9b-fp8 default, flux.2-klein-9b for full precision, flux.2-klein-4b).")
    parser.add_argument("--fast", action="store_true",
                        help="[pipeline] Use 4B model for much faster load (fewer weights than 9B).")

    # Shared SAM / VLM options
    parser.add_argument("--sam-model", type=str, default="facebook/sam3",
                        help="SAM 3 Hugging Face model id.")
    parser.add_argument("--sam-remove-threshold", type=float, default=0.4,
                        help="SAM confidence for objects/logos (lower = more sensitive).")
    parser.add_argument("--sam-retouch-threshold", type=float, default=0.65,
                        help="SAM confidence for surfaces/stains (higher = more certain).")
    parser.add_argument("--dilation", type=int, default=8,
                        help="Mask dilation in pixels.")
    parser.add_argument("--max-side", type=int, default=2048,
                        help="Max image dimension — large images downscaled at pipeline start (default: 2048).")
    parser.add_argument("--vlm-model", type=str, default="gemini-3.1-pro-preview",
                        help="Gemini VLM model.")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        sys.exit(f"Input directory not found: {input_dir}")

    pairs = find_paired_images(input_dir, target_dir, args.n)
    if not pairs:
        sys.exit(f"No images found in {input_dir}")
    print(f"Found {len(pairs)} image(s) ({sum(1 for p in pairs if p['target'])} with targets)\n")

    run_masks = args.test in ("masks", "all")
    run_pipeline = args.test in ("pipeline", "all")

    if run_masks:
        print("=" * 60)
        print("TEST: SAM Mask Accuracy")
        print("=" * 60)
        mask_output = output_dir / "mask_accuracy.jpg"
        test_masks(
            pairs, mask_output,
            use_vlm=args.use_vlm,
            vlm_model=args.vlm_model,
            sam_model=args.sam_model,
            sam_remove_threshold=args.sam_remove_threshold,
            sam_retouch_threshold=args.sam_retouch_threshold,
            dilation_px=args.dilation,
            mode=args.mode,
            overlay_alpha=args.overlay_alpha,
        )

    if run_pipeline:
        print("\n" + "=" * 60)
        print("TEST: Full Pipeline")
        print("=" * 60)
        pipeline_output = output_dir / "pipeline_evaluation.jpg"
        test_pipeline(
            pairs, pipeline_output,
            flux_model=args.flux_model,
            fast=args.fast,
            vlm_model=args.vlm_model,
            sam_model=args.sam_model,
            skip_vlm=args.skip_vlm,
            two_pass=not args.single_pass,
            remove_strength=args.remove_strength,
            retouch_strength=args.retouch_strength,
            num_steps=args.num_steps,
            guidance=args.guidance,
            seed=args.seed,
            sam_remove_threshold=args.sam_remove_threshold,
            sam_retouch_threshold=args.sam_retouch_threshold,
            dilation_px=args.dilation,
            max_side=args.max_side,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
