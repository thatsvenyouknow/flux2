"""
TUI Retouching Pipeline — end-to-end local inpainting for hotel/resort images.

Orchestrates:
  1. Gemini VLM analysis   → structured issues + edit prompt
  2. SAM 3 mask generation → per-category pixel masks
  3. Flux2Pipeline inpainting → local masked image editing

Supports single-pass (one combined mask) and two-pass (remove at high strength,
then retouch at lower strength) modes.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Maximum dimension (width or height) for pipeline input images.
# Images larger than this are downscaled (preserving aspect ratio) at the start
# of the pipeline so that VLM, SAM, and Flux all operate on a manageable size.
MAX_SIDE = 2048

from bertelsman.masking import (
    PROMPTS_REMOVE,
    PROMPTS_RETOUCH,
    MaskResult,
    SAM3Session,
    generate_masks,
    save_mask_debug,
)
from bertelsman.vlm import (
    DEFAULT_VLM_MODEL,
    VLMAnalysis,
    analyze_image,
    load_config,
)
from flux2.pipeline import Flux2Pipeline


@dataclass
class RetouchResult:
    """Output container for a retouching run."""
    image: Image.Image
    analysis: VLMAnalysis | None
    mask_result: MaskResult | None
    elapsed_seconds: float
    passes: list[str]


class TUIRetouchingPipeline:
    """
    Combines VLM analysis, SAM 3 masking, and Flux2 inpainting into a
    single retouching pipeline for TUI hotel/resort images.
    """

    def __init__(
        self,
        flux_pipeline: Flux2Pipeline,
        config_path: Path | str | None = None,
        sam_model_id: str = "facebook/sam3",
        vlm_model: str = DEFAULT_VLM_MODEL,
        sam_device: str | None = None,
    ):
        self.flux = flux_pipeline
        self.config = load_config(config_path)
        self.sam_model_id = sam_model_id
        self.vlm_model = vlm_model
        self.sam_device = sam_device

        # Load SAM once and keep on CPU between images
        self.sam_session = SAM3Session(model_id=sam_model_id)

    # ── Public API ────────────────────────────────────────────────────

    def run(
        self,
        image_path: Path | str,
        *,
        # VLM
        extra_task: str | None = None,
        skip_vlm: bool = False,
        manual_prompt: str | None = None,
        # Masking
        remove_prompts: list[str] | None = None,
        retouch_prompts: list[str] | None = None,
        dilation_px: int = 8,
        sam_remove_threshold: float = 0.35,
        sam_retouch_threshold: float = 0.65,
        # Inpainting
        two_pass: bool = False,
        remove_strength: float = 1.0,
        retouch_strength: float = 0.55,
        num_steps: int = 4,
        guidance: float = 4.0,
        seed: int | None = None,
        letterboxing: bool = True,
        # Image sizing
        max_side: int = MAX_SIDE,
        # Debug
        save_masks_dir: Path | str | None = None,
    ) -> RetouchResult:
        """
        Run the full retouching pipeline on a single image.

        Parameters
        ----------
        image_path : input image
        extra_task : additional instruction for the VLM
        skip_vlm : skip VLM analysis, use default/manual prompts directly
        manual_prompt : override edit prompt (skips VLM prompt generation)
        remove_prompts : override SAM prompts for remove category
        retouch_prompts : override SAM prompts for retouch category
        dilation_px : expand masks by this many pixels
        sam_remove_threshold : SAM confidence for objects (lower = more sensitive)
        sam_retouch_threshold : SAM confidence for surfaces (higher = more certain)
        two_pass : use separate remove/retouch passes with different strengths
        remove_strength : inpainting strength for remove pass
        retouch_strength : inpainting strength for retouch pass
        num_steps : Flux2 denoising steps
        guidance : Flux2 guidance scale
        seed : reproducibility seed
        letterboxing : pad image to multiple of 16
        max_side : max dimension (width or height) — images are downscaled at the
                   start of the pipeline so VLM, SAM, and Flux all work on the
                   same manageable resolution (default 2048)
        save_masks_dir : if set, save debug mask images here
        """
        image_path = Path(image_path)
        t0 = time.time()
        passes: list[str] = []

        # ── Downscale once for the entire pipeline ───────────────────
        image_path, _tmp_file = self._maybe_downscale(image_path, max_side)

        # ── Step 1: VLM analysis ──────────────────────────────────────
        analysis: VLMAnalysis | None = None
        if not skip_vlm and manual_prompt is None:
            print("[1/3] Analyzing image with Gemini VLM...")
            analysis = analyze_image(
                image_path,
                config=self.config,
                extra_task=extra_task,
                model=self.vlm_model,
            )
            print(f"  Scene: {analysis.scene_description}")
            print(f"  Issues: {len(analysis.issues)} "
                  f"({len(analysis.remove_issues)} remove, "
                  f"{len(analysis.retouch_issues)} retouch)")
            if analysis.tui_logos_present:
                print(f"  TUI logos (preserved): {analysis.tui_logos_present}")
            print(f"  Edit prompt: {analysis.edit_prompt}")
        else:
            print("[1/3] Skipping VLM analysis.")

        # Resolve edit prompt — use remove-only variant when retouch is disabled
        if manual_prompt:
            edit_prompt = manual_prompt
        elif analysis is not None:
            edit_prompt = analysis.remove_only_edit_prompt or analysis.edit_prompt
        else:
            edit_prompt = ""
        if not edit_prompt:
            edit_prompt = "Maintain all aspects of the original image exactly as-is."

        # Reinforce inpainting behaviour: fill masked areas with surrounding
        # background, never invent new objects or leave shadows/artifacts.
        _INPAINT_SUFFIX = (
            "Fill every removed area exclusively with the surrounding "
            "background texture. Do not introduce any new objects, shadows, "
            "or artifacts in the filled regions."
        )
        if "Maintain all aspects" not in edit_prompt:
            edit_prompt = edit_prompt.rstrip(". ") + ". " + _INPAINT_SUFFIX

        # ── Step 2: SAM 3 mask generation ─────────────────────────────
        print("[2/3] Generating masks with SAM 3...")

        # Build SAM prompts: use VLM results when available (even if empty —
        # an empty list means VLM found nothing to fix).  Fall back to default
        # prompt lists only when VLM was skipped / not used.
        if remove_prompts is None:
            if analysis is not None:
                remove_prompts = analysis.remove_sam_prompts
            else:
                remove_prompts = list(PROMPTS_REMOVE)

        # TODO: re-enable retouch pass once remove quality is validated
        # if retouch_prompts is None:
        #     if analysis is not None:
        #         retouch_prompts = analysis.retouch_sam_prompts
        #     else:
        #         retouch_prompts = list(PROMPTS_RETOUCH)
        retouch_prompts = []

        mask_result = generate_masks(
            image_path,
            remove_prompts=remove_prompts,
            retouch_prompts=retouch_prompts,
            model_id=self.sam_model_id,
            remove_threshold=sam_remove_threshold,
            retouch_threshold=sam_retouch_threshold,
            dilation_px=dilation_px,
            device=self.sam_device,
            session=self.sam_session,
        )

        if save_masks_dir:
            saved = save_mask_debug(mask_result, image_path, save_masks_dir)
            print(f"  Saved {len(saved)} debug mask(s) to {save_masks_dir}")

        # ── Step 3: Flux2 inpainting ──────────────────────────────────
        print("[3/3] Inpainting with Flux2...")

        has_remove = mask_result.remove_combined is not None and mask_result.remove_combined.any()
        has_retouch = mask_result.retouch_combined is not None and mask_result.retouch_combined.any()

        if not has_remove and not has_retouch:
            print("  No masks detected — returning original image.")
            result_img = Image.open(image_path).convert("RGB")
            return RetouchResult(
                image=result_img,
                analysis=analysis,
                mask_result=mask_result,
                elapsed_seconds=time.time() - t0,
                passes=["no-op"],
            )

        # TODO: re-enable two-pass once retouch quality is validated
        # if two_pass and has_remove and has_retouch:
        #     result_img = self._two_pass_inpaint(
        #         image_path=image_path,
        #         mask_result=mask_result,
        #         edit_prompt=edit_prompt,
        #         remove_strength=remove_strength,
        #         retouch_strength=retouch_strength,
        #         num_steps=num_steps,
        #         guidance=guidance,
        #         seed=seed,
        #         letterboxing=letterboxing,
        #     )
        #     passes = ["remove", "retouch"]

        # Single remove pass at full strength
        strength = remove_strength if has_remove else retouch_strength
        combined = mask_result.all_combined
        mask_pil = Image.fromarray(combined)
        result_img = self._inpaint_once(
            input_image=image_path,
            mask=mask_pil,
            prompt=edit_prompt,
            strength=strength,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            letterboxing=letterboxing,
        )
        passes = ["remove"]

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({', '.join(passes)} pass{'es' if len(passes) > 1 else ''})")

        return RetouchResult(
            image=result_img,
            analysis=analysis,
            mask_result=mask_result,
            elapsed_seconds=elapsed,
            passes=passes,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _maybe_downscale(
        image_path: Path, max_side: int,
    ) -> tuple[Path, tempfile.NamedTemporaryFile | None]:
        """Downscale image if either side exceeds *max_side*, preserving aspect ratio.

        Returns the (possibly new) path and a temp-file handle that the caller
        must keep alive until the path is no longer needed.
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest <= max_side:
            return image_path, None

        scale = max_side / longest
        new_w = int(w * scale) // 16 * 16 or 16
        new_h = int(h * scale) // 16 * 16 or 16
        print(f"  Downscaling input: {w}×{h} → {new_w}×{new_h} (max_side={max_side})")
        img = img.resize((new_w, new_h), Image.LANCZOS)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=True)
        img.save(tmp, format="JPEG", quality=95, subsampling=0)
        tmp.flush()
        return Path(tmp.name), tmp

    def _inpaint_once(
        self,
        input_image: Path | Image.Image,
        mask: Image.Image,
        prompt: str,
        strength: float,
        num_steps: int,
        guidance: float,
        seed: int | None,
        letterboxing: bool,
    ) -> Image.Image:
        """Single inpainting pass through Flux2Pipeline."""
        return self.flux.generate(
            prompt=prompt,
            input_image=input_image,
            inpainting_mask=mask,
            strength=strength,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            letterboxing=letterboxing,
            i2i_mode="inpainting",
        )

    def _two_pass_inpaint(
        self,
        image_path: Path,
        mask_result: MaskResult,
        edit_prompt: str,
        remove_strength: float,
        retouch_strength: float,
        num_steps: int,
        guidance: float,
        seed: int | None,
        letterboxing: bool,
    ) -> Image.Image:
        """Two-pass strategy: remove objects first, then retouch surfaces."""

        # Pass 1: Remove (high strength)
        remove_mask_pil = Image.fromarray(mask_result.remove_combined)
        print(f"  Pass 1/2: REMOVE (strength={remove_strength})")
        intermediate = self._inpaint_once(
            input_image=image_path,
            mask=remove_mask_pil,
            prompt=edit_prompt,
            strength=remove_strength,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            letterboxing=letterboxing,
        )

        # Pass 2: Retouch (lower strength) — feed intermediate result
        retouch_mask_pil = Image.fromarray(mask_result.retouch_combined)
        print(f"  Pass 2/2: RETOUCH (strength={retouch_strength})")
        result = self._inpaint_once(
            input_image=intermediate,
            mask=retouch_mask_pil,
            prompt=edit_prompt,
            strength=retouch_strength,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            letterboxing=letterboxing,
        )

        return result


# ── Utility: save run log ─────────────────────────────────────────────────

def save_run_log(
    image_path: Path,
    output_path: Path,
    result: RetouchResult,
) -> Path | None:
    """Write a JSON log for auditing/debugging. Returns log path or None."""
    log_dir = output_path.parent / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{output_path.stem}.json"
        data = {
            "input": str(image_path),
            "output": str(output_path),
            "prompt": result.analysis.edit_prompt if result.analysis else "",
            "passes": result.passes,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "timestamp": datetime.now().isoformat(),
        }
        if result.analysis:
            data["scene_type"] = result.analysis.scene_type
            data["num_issues"] = len(result.analysis.issues)
            data["tui_logos"] = result.analysis.tui_logos_present
        log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return log_path
    except (PermissionError, OSError):
        return None
