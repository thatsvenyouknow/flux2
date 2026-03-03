"""
Gemini VLM image analysis for TUI hotel/resort retouching.

Sends an image + the retouching system prompt to Gemini, parses the structured
JSON response, and categorises each detected issue as REMOVE or RETOUCH so
downstream mask generation / inpainting can use the right strategy.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PKG_ROOT = Path(__file__).parent

# Load .env from flux2 project root (parent of src/) or current working directory
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Prefer flux2 root: .../flux2/src/bertelsman/vlm.py -> .../flux2
    flux2_root = PKG_ROOT.parent.parent
    for path in (flux2_root / ".env", Path.cwd() / ".env"):
        if path.exists():
            load_dotenv(path)
            break


_load_dotenv()

DEFAULT_CONFIG = PKG_ROOT / "config" / "retouching_config.yaml"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

REMOVE_CATEGORIES = frozenset({
    "logo_brand",
    "equipment_clutter",
    "safety_devices",
    "infrastructure",
    "stickers",
    "reflections",
    "construction",
    "damage_dirt",
    "water_stains",
    "wear_damage",  # legacy alias — VLM may still produce this
})

RETOUCH_CATEGORIES = frozenset({
    "linen_textile",
    "surface_clean",
    "vegetation",
    "skin_retouch",
    "wood_repair",
})


@dataclass
class Issue:
    category: str
    description: str
    sam_label: str = ""
    location: str = ""
    severity: str = "moderate"
    confidence: float = 50.0

    @property
    def is_remove(self) -> bool:
        return self.category in REMOVE_CATEGORIES

    @property
    def is_retouch(self) -> bool:
        return self.category in RETOUCH_CATEGORIES

    @property
    def sam_prompt(self) -> str:
        """Short object noun for SAM. Prefers the dedicated sam_label; falls back to description."""
        return self.sam_label.strip() if self.sam_label.strip() else self.description


@dataclass
class VLMAnalysis:
    scene_description: str = ""
    scene_type: str = "other"
    tui_logos_present: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    edit_prompt: str = ""
    overall_quality: str = "good"

    @property
    def remove_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.is_remove]

    @property
    def retouch_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.is_retouch]

    @property
    def remove_sam_prompts(self) -> list[str]:
        """Short SAM-friendly labels for objects to remove."""
        return [i.sam_prompt for i in self.remove_issues]

    @property
    def retouch_sam_prompts(self) -> list[str]:
        """Short SAM-friendly labels for items to retouch."""
        return [i.sam_prompt for i in self.retouch_issues]

    @property
    def remove_descriptions(self) -> list[str]:
        """Full descriptions for remove issues (for logging / debug)."""
        return [i.description for i in self.remove_issues]

    @property
    def retouch_descriptions(self) -> list[str]:
        """Full descriptions for retouch issues (for logging / debug)."""
        return [i.description for i in self.retouch_issues]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path | str | None = None) -> dict:
    """Load retouching YAML config. Returns dict with at least 'system_prompt'."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path) as f:
        raw = f.read()
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass
    return {"system_prompt": raw}


def _load_image_as_base64(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


def _parse_vlm_json(raw_text: str) -> VLMAnalysis:
    """Parse the structured JSON returned by Gemini into a VLMAnalysis."""
    text = _strip_code_fences(raw_text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return VLMAnalysis(edit_prompt=raw_text.strip())

    if not isinstance(data, dict):
        return VLMAnalysis(edit_prompt=raw_text.strip())

    issues: list[Issue] = []
    for raw_issue in data.get("issues", []):
        if isinstance(raw_issue, dict):
            issues.append(Issue(
                category=raw_issue.get("category", ""),
                description=raw_issue.get("description", ""),
                sam_label=raw_issue.get("sam_label", ""),
                location=raw_issue.get("location", ""),
                severity=raw_issue.get("severity", "moderate"),
                confidence=float(raw_issue.get("confidence", 50)),
            ))

    return VLMAnalysis(
        scene_description=data.get("scene_description", ""),
        scene_type=data.get("scene_type", "other"),
        tui_logos_present=data.get("tui_logos_present", []),
        issues=issues,
        edit_prompt=data.get("edit_prompt", ""),
        overall_quality=data.get("overall_quality", "good"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_VLM_MODEL = "gemini-2.5-pro"


def analyze_image(
    image_path: Path | str,
    config: dict | None = None,
    extra_task: str | None = None,
    model: str = DEFAULT_VLM_MODEL,
    api_key: str | None = None,
) -> VLMAnalysis:
    """
    Analyze an image with Gemini and return structured retouching analysis.

    Parameters
    ----------
    image_path : path to the input image
    config : loaded YAML config dict (must contain 'system_prompt' key).
             Defaults to the bundled retouching_config.yaml.
    extra_task : optional additional instruction appended to the user message
    model : Gemini model name
    api_key : Gemini API key (falls back to GEMINI_API_KEY env var)
    """
    try:
        from google import genai
    except ImportError:
        sys.exit("Install google-genai: pip install google-genai")

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set. Add it to .env or export it.")

    if config is None:
        config = load_config()
    system_prompt = config.get("system_prompt", "")
    if isinstance(system_prompt, dict):
        system_prompt = str(system_prompt)

    image_path = Path(image_path)
    b64, mime = _load_image_as_base64(image_path)

    client = genai.Client(api_key=key)
    image_part = genai.types.Part.from_bytes(
        data=base64.b64decode(b64), mime_type=mime
    )

    user_message = "Analyze this image and write the retouching/editing prompt."
    if extra_task:
        user_message += f"\nAdditional instruction: {extra_task}"

    response = client.models.generate_content(
        model=model,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.0,
            seed=42,
        ),
        contents=[image_part, user_message],
    )

    return _parse_vlm_json(response.text)
