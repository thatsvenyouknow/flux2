"""
SAM 3 mask generation for TUI retouching pipeline.

Generates pixel-level segmentation masks from text prompts using SAM 3
(facebook/sam3 via Hugging Face transformers).  Prompts can come dynamically
from VLM-detected issues or from the default concept lists.

Supports returning separate masks for REMOVE vs RETOUCH categories so the
downstream inpainting pipeline can use different strengths per pass.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Default concept lists ──────────────────────────────────────────────────
# REMOVE = discrete objects that should be fully replaced with background.
# RETOUCH = surface conditions / textures that need subtle cleanup.

PROMPTS_REMOVE = [
    # Logos & branding
    "logo",
    "brand name",
    "text on clothing",
    "text on bag",
    "alcohol bottle label",
    "book title",
    "magazine cover",
    # Construction & maintenance
    "construction debris",
    "scaffolding",
    "construction equipment",
    # Safety & utility equipment
    "fire extinguisher",
    "fire hose",
    "fire alarm",
    "safety sign",
    "warning sign",
    "life buoy",
    "first aid kit",
    "cleaning cart",
    "mop bucket",
    # Outdoor equipment
    "garden hose",
    "air conditioning unit",
    "satellite dish",
    # Safety devices
    "smoke detector",
    "surveillance camera",
    "security camera",
    "motion sensor",
    # Infrastructure
    "power line",
    "electricity pole",
    "electrical cable",
    "pool jet",
    "pool outlet",
    "drain cover",
    "utility box",
    # Structural damage
    "crack in wall",
    "cracked tile",
    "broken tile",
    "broken fixture",
    "chipped countertop",
    # Stickers & reflections
    "window sticker",
    "sticker on glass",
    "reflection of camera",
    "reflection of photographer",
    # Trash & clutter
    "trash bin",
    "ashtray",
    "crumbs on floor",
    # Dirt, damage & stains (visually distracting)
    "rust",
    "rust spot",
    "dirt",
    "stain",
    "water stain",
    "limescale",
    "mineral deposit",
    "discoloration",
    "peeling paint",
    "chipped paint",
    "debris",
]

PROMPTS_RETOUCH = [
    "wrinkled bedding",
    "messy pillows",
    "rumpled tablecloth",
    "dirty grout",
    "footprints in sand",
    "messy sand",
    "burnt grass",
    "dead palm fronds",
    "scratched wood furniture",
    "water ring on wood",
]

# Objects that belong in hotel/resort scenes and must NEVER be masked,
# even if a VLM issue description mentions them.  Matched case-insensitively
# as substrings against SAM prompt text.
# With the sam_label approach the VLM produces short object nouns for SAM
# (e.g. "logo", "crack", "limescale"), so blocklist matches are rare.  We keep
# this as a safety net against VLM hallucinations that name whole scene objects.
SAM_BLOCKLIST = [
    "picture",
    "towel",
    "lounger",
    "sun lounger",
    "beach umbrella",
    "parasol",
    "pool float",
    "pool noodle",
    "boat",
    "sailboat",
    "yacht",
    "jet ski",
    "kayak",
    "car",
    "bus",
    "food",
    "drink",
    "plate",
    "person",
    "guest",
    "staff",
]


@dataclass
class SegmentMask:
    label: str
    mask: np.ndarray  # (H, W) uint8  0/255


@dataclass
class MaskResult:
    """Container for categorised mask outputs."""
    remove_masks: list[SegmentMask] = field(default_factory=list)
    retouch_masks: list[SegmentMask] = field(default_factory=list)
    remove_combined: np.ndarray | None = None   # (H, W) uint8 union
    retouch_combined: np.ndarray | None = None   # (H, W) uint8 union
    all_combined: np.ndarray | None = None       # (H, W) uint8 union of both


# ── Helpers ────────────────────────────────────────────────────────────────

def _mask_to_uint8(m) -> np.ndarray:
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    m = np.asarray(m).squeeze()
    return ((m > 0.5) if m.dtype in (np.float32, np.float64) else (m > 0)).astype(np.uint8) * 255


def _combine_masks(masks: list[SegmentMask], h: int, w: int) -> np.ndarray:
    if not masks:
        return np.zeros((h, w), dtype=np.uint8)
    stacked = np.stack([m.mask for m in masks], axis=0)
    return np.clip(stacked.max(axis=0), 0, 255).astype(np.uint8)


def _filter_blocklist(prompts: list[str], blocklist: list[str] = SAM_BLOCKLIST) -> list[str]:
    """Remove prompts that mention blocklisted objects (case-insensitive substring match)."""
    filtered = []
    blocked = []
    for prompt in prompts:
        lower = prompt.lower()
        if any(term in lower for term in blocklist):
            blocked.append(prompt)
        else:
            filtered.append(prompt)
    if blocked:
        print(f"  SAM blocklist filtered {len(blocked)} prompt(s): {blocked}")
    return filtered


def _dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """Dilate a binary mask to expand coverage around detected objects."""
    if dilation_px <= 0:
        return mask
    from PIL import Image, ImageFilter
    pil = Image.fromarray(mask)
    pil = pil.filter(ImageFilter.MaxFilter(size=2 * dilation_px + 1))
    return np.array(pil)


# ── Persistent SAM 3 session ──────────────────────────────────────────────

class SAM3Session:
    """Loads SAM 3 once and keeps it on CPU between inference calls.

    On ``activate()`` the model is moved to GPU, used for mask extraction,
    then ``deactivate()`` moves it back to CPU so VRAM is available for other
    models (Flux2, AE).  Across multiple images the expensive
    ``from_pretrained`` call happens only once.
    """

    def __init__(self, model_id: str = "facebook/sam3"):
        try:
            from transformers import Sam3Model, Sam3Processor
            import torch  # noqa: F811
        except ImportError as e:
            if "Sam3Model" in str(e) or "Sam3Processor" in str(e):
                sys.exit(
                    "SAM 3 requires transformers >= 4.57. Upgrade with:\n"
                    "  pip install 'transformers>=4.57'\n"
                    "Then ensure you are logged in: huggingface-cli login\n"
                    "  https://huggingface.co/facebook/sam3"
                )
            sys.exit(
                f"Missing dependency for SAM 3: {e}\n"
                "Install: pip install transformers torch pillow\n"
                "Then: huggingface-cli login; https://huggingface.co/facebook/sam3"
            )

        print(f"  Loading SAM 3 ({model_id}) → CPU...")
        self.model = Sam3Model.from_pretrained(model_id)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self._on_gpu = False

    # ── GPU lifecycle ─────────────────────────────────────────────

    def activate(self, device: str = "cuda") -> None:
        """Move SAM to *device* (GPU) for inference."""
        if not self._on_gpu:
            self.model.to(device)
            self._on_gpu = True
            self._device = device

    def deactivate(self) -> None:
        """Move SAM back to CPU and free GPU memory."""
        import torch
        if self._on_gpu:
            self.model.cpu()
            self._on_gpu = False
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────

    def extract_masks(
        self,
        image_path: Path,
        text_prompts: list[str],
        threshold: float,
        mask_threshold: float,
    ) -> list[SegmentMask]:
        """Run SAM 3 on *text_prompts* for a single image.

        The model must already be on GPU (call ``activate()`` first).
        Vision embeddings are computed once and shared across all prompts.
        """
        import torch
        from PIL import Image as PILImage

        device = self._device

        image = PILImage.open(image_path).convert("RGB")
        h, w = image.height, image.width

        img_inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(
                pixel_values=img_inputs.pixel_values,
            )

        all_masks: list[SegmentMask] = []
        for prompt in text_prompts:
            text_inputs = self.processor(text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=img_inputs.get("original_sizes").tolist(),
            )[0]
            masks = results.get("masks")
            if masks is None:
                continue
            if isinstance(masks, (list, tuple)):
                for m in masks:
                    mask_np = _mask_to_uint8(m)
                    if mask_np.shape[0] != h or mask_np.shape[1] != w:
                        mask_np = np.array(PILImage.fromarray(mask_np).resize((w, h), PILImage.NEAREST))
                    all_masks.append(SegmentMask(label=prompt, mask=mask_np))
            else:
                for i in range(masks.shape[0]):
                    mask_np = _mask_to_uint8(masks[i])
                    if mask_np.shape[0] != h or mask_np.shape[1] != w:
                        mask_np = np.array(PILImage.fromarray(mask_np).resize((w, h), PILImage.NEAREST))
                    all_masks.append(SegmentMask(label=prompt, mask=mask_np))

        del vision_embeds, img_inputs
        return all_masks


# ── Public API ────────────────────────────────────────────────────────────

# Per-category SAM confidence defaults.
# Objects (logos, devices) are visually distinct → lower threshold OK.
# Subtle surface issues (stains, dirt) need high certainty → higher threshold.
DEFAULT_REMOVE_THRESHOLD = 0.35
DEFAULT_RETOUCH_THRESHOLD = 0.65


def generate_masks(
    image_path: Path | str,
    remove_prompts: list[str] | None = None,
    retouch_prompts: list[str] | None = None,
    model_id: str = "facebook/sam3",
    remove_threshold: float = DEFAULT_REMOVE_THRESHOLD,
    retouch_threshold: float = DEFAULT_RETOUCH_THRESHOLD,
    dilation_px: int = 8,
    device: str | None = None,
    session: SAM3Session | None = None,
) -> MaskResult:
    """
    Generate categorised segmentation masks for an image.

    Parameters
    ----------
    image_path : path to the input image
    remove_prompts : text prompts for objects to remove (default: PROMPTS_REMOVE)
    retouch_prompts : text prompts for items to retouch (default: PROMPTS_RETOUCH)
    model_id : Hugging Face SAM 3 model id
    remove_threshold : SAM confidence for remove prompts (lower = more sensitive)
    retouch_threshold : SAM confidence for retouch prompts (higher = more certain)
    dilation_px : expand masks by this many pixels for better inpainting coverage
    device : "cuda", "cpu", or None (auto)
    session : reusable SAM3Session (avoids reloading weights from disk each call)
    """
    import torch

    image_path = Path(image_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    remove_prompts = remove_prompts if remove_prompts is not None else list(PROMPTS_REMOVE)
    retouch_prompts = retouch_prompts if retouch_prompts is not None else list(PROMPTS_RETOUCH)

    remove_prompts = _filter_blocklist(remove_prompts)
    retouch_prompts = _filter_blocklist(retouch_prompts)

    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    h, w = img.height, img.width

    if not remove_prompts and not retouch_prompts:
        return MaskResult(
            remove_combined=np.zeros((h, w), dtype=np.uint8),
            retouch_combined=np.zeros((h, w), dtype=np.uint8),
            all_combined=np.zeros((h, w), dtype=np.uint8),
        )

    # Use a persistent session if provided, otherwise create a temporary one.
    owns_session = session is None
    if owns_session:
        session = SAM3Session(model_id=model_id)

    session.activate(device)

    remove_masks: list[SegmentMask] = []
    retouch_masks: list[SegmentMask] = []

    if remove_prompts:
        print(f"  SAM remove ({len(remove_prompts)} prompts, threshold={remove_threshold})...")
        remove_masks = session.extract_masks(
            image_path, remove_prompts,
            threshold=remove_threshold, mask_threshold=remove_threshold,
        )

    if retouch_prompts:
        print(f"  SAM retouch ({len(retouch_prompts)} prompts, threshold={retouch_threshold})...")
        retouch_masks = session.extract_masks(
            image_path, retouch_prompts,
            threshold=retouch_threshold, mask_threshold=retouch_threshold,
        )

    # Move SAM back to CPU (or free entirely if we created a throwaway session)
    if owns_session:
        del session
        torch.cuda.empty_cache()
    else:
        session.deactivate()

    remove_combined = _combine_masks(remove_masks, h, w)
    retouch_combined = _combine_masks(retouch_masks, h, w)

    if dilation_px > 0:
        remove_combined = _dilate_mask(remove_combined, dilation_px)
        retouch_combined = _dilate_mask(retouch_combined, dilation_px)
        for m in remove_masks:
            m.mask = _dilate_mask(m.mask, dilation_px)
        for m in retouch_masks:
            m.mask = _dilate_mask(m.mask, dilation_px)

    all_combined = np.clip(
        np.maximum(remove_combined, retouch_combined), 0, 255,
    ).astype(np.uint8)

    print(f"  SAM masks: {len(remove_masks)} remove, {len(retouch_masks)} retouch "
          f"(dilation={dilation_px}px)")

    return MaskResult(
        remove_masks=remove_masks,
        retouch_masks=retouch_masks,
        remove_combined=remove_combined,
        retouch_combined=retouch_combined,
        all_combined=all_combined,
    )


def save_mask_debug(
    mask_result: MaskResult,
    image_path: Path | str,
    output_dir: Path | str,
) -> list[Path]:
    """Save mask images for visual inspection. Returns list of saved paths."""
    from PIL import Image as PILImage

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    saved: list[Path] = []

    for name, arr in [
        ("remove", mask_result.remove_combined),
        ("retouch", mask_result.retouch_combined),
        ("all", mask_result.all_combined),
    ]:
        if arr is not None:
            p = output_dir / f"{stem}_mask_{name}.png"
            PILImage.fromarray(arr).save(p)
            saved.append(p)

    for i, seg in enumerate(mask_result.remove_masks + mask_result.retouch_masks):
        slug = seg.label.replace(" ", "_")[:20]
        p = output_dir / f"{stem}_mask_{i:02d}_{slug}.png"
        PILImage.fromarray(seg.mask).save(p)
        saved.append(p)

    return saved
