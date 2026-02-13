import json
import os
import random
import shlex
import sys
import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from einops import rearrange
from PIL import ExifTags, Image

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    center_crop_to_multiple_of_x,
    default_prep,
    denoise,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

# from flux2.watermark import embed_watermark


@dataclass
class Config:
    prompt: str = "Turn the image into a professional logo shot in a sci-fi setting."
    seed: Optional[int] = None
    width: int = 1360
    height: int = 768
    num_steps: int = 50
    guidance: float = 4.0
    input_image: Optional[Path] = None
    cond_images: List[Path] = field(default_factory=list)
    match_image_size: Optional[int] = None  # Index of cond_images to match size from
    upsample_prompt_mode: Literal["none", "local"] = "none"
    letterboxing: bool = False
    letterboxing_color: str = "#000000"
    i2i_mode: Literal["none", "inpainting", "outpainting"] = "none"
    inpainting_mask: Optional[Path] = None
    strength: float = 1.0  # 0.0 = no change, 1.0 = full regeneration

    def copy(self) -> "Config":
        return Config(
            prompt=self.prompt,
            seed=self.seed,
            width=self.width,
            height=self.height,
            num_steps=self.num_steps,
            guidance=self.guidance,
            input_image=self.input_image, #I2I
            cond_images=list(self.cond_images),
            match_image_size=self.match_image_size,
            upsample_prompt_mode=self.upsample_prompt_mode,
            letterboxing=self.letterboxing,
            letterboxing_color=self.letterboxing_color,
            i2i_mode=self.i2i_mode,
            inpainting_mask=self.inpainting_mask,
            strength=self.strength,
        )

INT_FIELDS = {"width", "height", "seed", "num_steps", "match_image_size"}
BOOL_FIELDS = {"letterboxing"}
FLOAT_FIELDS = {"guidance", "strength"}
LIST_FIELDS = {"cond_images"}
UPSAMPLING_MODE_FIELDS = {"none", "local"}
STR_FIELDS = {"letterboxing_color"}
I2I_MODE_FIELDS = {"none", "inpainting", "outpainting"}
PATH_FIELDS = {"input_image", "inpainting_mask"}


def print_config(cfg: Config):
    d = asdict(cfg)
    d["cond_images"] = [str(p) for p in cfg.cond_images]
    print("Current config:")
    for k in [
        "prompt",
        "seed",
        "width",
        "height",
        "num_steps",
        "guidance",
        "input_image",
        "cond_images",
        "match_image_size",
        "upsample_prompt_mode",
        "i2i_mode",
        "inpainting_mask",
        "strength",
        "letterboxing",
        "letterboxing_color",
    ]:
        print(f"  {k}: {d[k]}")
    print()


def validate_model_params(model_name: str, cfg: Config) -> bool:
    """Validate that config parameters match model requirements. Returns True if valid."""
    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info.get("defaults", {})
    fixed_params = model_info.get("fixed_params", set())

    errors = []
    if "num_steps" in fixed_params and cfg.num_steps != defaults["num_steps"]:
        errors.append(
            f"Model '{model_name}' requires num_steps={defaults['num_steps']}, "
            f"but you specified num_steps={cfg.num_steps}"
        )

    if "guidance" in fixed_params and cfg.guidance != defaults["guidance"]:
        errors.append(
            f"Model '{model_name}' requires guidance={defaults['guidance']}, "
            f"but you specified guidance={cfg.guidance}"
        )

    if errors:
        print("\nERROR: Invalid parameters for selected model:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nPlease adjust your parameters and try again.", file=sys.stderr)
        return False

    return True


def letterbox_to_multiple_of_x(img: Image.Image, x: int, color: str = "#000000") -> Image.Image:
    """
    Letterbox the image to a multiple of x.
    """

    w, h = img.size
    new_w = (w + x - 1) // x * x
    new_h = (h + x - 1) // x * x
   
    #No letterboxing needed
    if new_w == w and new_h == h:
        return img
    
    #Create new image with the new size and letterbox color
    new_img = Image.new("RGB", (new_w, new_h), color)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    new_img.paste(img, (left, top))
    return new_img

# ---------- Main Loop ----------


def main(
    model_name: str = "flux.2-klein-4b",
    prompt: str | None = None,
    debug_mode: bool = False,
    cpu_offloading: bool = True,
):
    model_info = FLUX2_MODEL_INFO[model_name]
    torch_device = torch.device("cuda")
    
    #Load models
    text_encoder = load_text_encoder(model_name, device=torch_device)
    model = load_flow_model(
        model_name, debug_mode=debug_mode, device="cpu" if cpu_offloading else torch_device
    )
    ae = load_ae(model_name)
    ae.eval()
    text_encoder.eval()

    #Optional: Upsample text prompt
    # if "klein" in model_name:
    #     mod_and_upsampling_model = load_text_encoder("flux.2-dev")
    # else:
    #     mod_and_upsampling_model = text_encoder
    mod_and_upsampling_model = text_encoder

    DEFAULTS = Config(
        prompt="Image of an anime character on a pirate ship", #"Make the character smile with a big happy grin, anime style",
        input_image=None, #Path("/home/sanctuary/projects/flux2/input_images/luffy_non16.jpg"),
        inpainting_mask=None, #Path("/home/sanctuary/projects/flux2/input_images/luffy_mouth_mask_small_non16.png"),
        i2i_mode="none", #"inpainting",
        strength=0.85,
        seed=42,
        letterboxing=False,
        letterboxing_color="#000000",
    )

    #Create and validate config
    cfg = DEFAULTS.copy()

    # Apply model defaults if not overridden
    defaults = model_info.get("defaults", {})
    if "num_steps" in defaults and "num_steps" not in overwrite:
        cfg.num_steps = defaults["num_steps"]
    if "guidance" in defaults and "guidance" not in overwrite:
        cfg.guidance = defaults["guidance"]

    if prompt is not None:
        cfg.prompt = prompt

    # Validate initial config
    if not validate_model_params(model_name, cfg):
        sys.exit(1)
    print_config(cfg)

    try:
        # Load explicit conditioning images first
        img_ctx = [Image.open(cond_image) for cond_image in cfg.cond_images]
        orig_width = None
        orig_height = None

        # Apply match_image_size if specified
        width = cfg.width
        height = cfg.height
        if cfg.match_image_size is not None:
            if cfg.match_image_size < 0 or cfg.match_image_size >= len(img_ctx):
                print(
                    f"  ! match_image_size={cfg.match_image_size} is out of range (0-{len(img_ctx)-1})",
                    file=sys.stderr,
                )
                print(f"  ! Using default dimensions: {width}x{height}", file=sys.stderr)
            else:
                ref_img = img_ctx[cfg.match_image_size]
                width, height = ref_img.size
                print(f"  Matched dimensions from image {cfg.match_image_size}: {width}x{height}")

        dir = Path("output")
        dir.mkdir(exist_ok=True)
        output_name = dir / f"sample_{len(list(dir.glob('*')))}.png"

        with torch.no_grad():
            ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

            if cfg.upsample_prompt_mode == "local":
                upsampled_prompts = mod_and_upsampling_model.upsample_prompt(
                    [cfg.prompt], img=[img_ctx] if img_ctx else None
                )
                prompt = upsampled_prompts[0] if upsampled_prompts else cfg.prompt
            else:
                prompt = cfg.prompt

            print("Generating with prompt: ", prompt)

            if model_info["guidance_distilled"]:
                ctx = text_encoder([prompt]).to(torch.bfloat16)
            else:
                ctx_empty = text_encoder([""]).to(torch.bfloat16)
                ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
                ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
            ctx, ctx_ids = batched_prc_txt(ctx)

            if cpu_offloading:
                text_encoder = text_encoder.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)
                if "klein" in model_name:
                    mod_and_upsampling_model = mod_and_upsampling_model.cpu()

            #I2I placeholder
            inpaint_mask_seq = None
            orig_img_seq = None
            noise_seq = None

            if cfg.input_image:
                input_img = Image.open(cfg.input_image).convert("RGB")
                orig_width, orig_height = input_img.size

                # Load inpainting mask (if provided) before any spatial transforms
                mask_pil = None
                if cfg.inpainting_mask:
                    mask_pil = Image.open(cfg.inpainting_mask).convert("L")
                    if mask_pil.size != input_img.size:
                        mask_pil = mask_pil.resize(input_img.size, Image.NEAREST)

                # Letterbox both image and mask together
                if cfg.letterboxing:
                    letterboxed_img = letterbox_to_multiple_of_x(
                        input_img, 16, color=cfg.letterboxing_color
                    )
                    if mask_pil is not None:
                        # Letterbox mask: pad with 0 (keep original) in border areas
                        lb_w, lb_h = letterboxed_img.size
                        mask_letterboxed = Image.new("L", (lb_w, lb_h), 0)
                        left = (lb_w - input_img.width) // 2
                        top = (lb_h - input_img.height) // 2
                        mask_letterboxed.paste(mask_pil, (left, top))
                        mask_pil = mask_letterboxed
                    input_img = letterboxed_img

                #Preprocess and encode clean input image 
                input_tensor = default_prep(input_img, limit_pixels=None, ensure_multiple=16)
                x_img = ae.encode(input_tensor[None].to(torch_device))[0].unsqueeze(0).to(torch.bfloat16)

                # Use actual latent-backed output size.
                _, _, latent_h, latent_w = x_img.shape
                width = latent_w * 16
                height = latent_h * 16

                # Generate noise in latent space
                generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
                noise_latent = torch.randn(
                    x_img.shape, generator=generator, dtype=torch.bfloat16, device="cuda"
                )

                # Convert clean latents and noise to sequence format
                x_clean_seq, x_ids = batched_prc_img(x_img)
                noise_seq_full, _ = batched_prc_img(noise_latent)

                # Compute full timestep schedule, then truncate based on strength
                full_timesteps = get_schedule(cfg.num_steps, x_clean_seq.shape[1])
                num_i2i_steps = max(1, int(cfg.num_steps * cfg.strength))
                timesteps = full_timesteps[-(num_i2i_steps + 1):]
                t_start = timesteps[0]
                print(f"  img2img: strength={cfg.strength}, steps={num_i2i_steps}/{cfg.num_steps}, t_start={t_start:.4f}")

                # Create noisy latents at t_start: x_t = (1-t)*x_0 + t*noise
                x = (1 - t_start) * x_clean_seq + t_start * noise_seq_full

                # Prepare inpainting mask in latent sequence format
                if mask_pil is not None:
                    # Apply same center crop as default_prep to keep mask aligned
                    mask_pil = center_crop_to_multiple_of_x(mask_pil, 16)
                    mask_np = np.array(mask_pil)
                    mask_bin = (mask_np > 0).astype(np.float32)
                    mask_tensor = torch.from_numpy(mask_bin).float()

                    # Downsample mask to latent resolution
                    mask_latent = torch.nn.functional.interpolate(
                        mask_tensor.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                        size=(latent_h, latent_w),
                        mode="nearest",
                    ).to(torch.bfloat16).to(torch_device)

                    # Flatten to sequence format: (1, latent_h*latent_w, 1)
                    inpaint_mask_seq = rearrange(mask_latent[0], "c h w -> (h w) c").unsqueeze(0)
                    orig_img_seq = x_clean_seq
                    noise_seq = noise_seq_full
                    print(f"  inpainting mask: {mask_bin.sum():.0f}/{mask_bin.size} latent pixels masked")
            else:
                shape = (1, 128, height // 16, width // 16)
                generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
                randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
                x, x_ids = batched_prc_img(randn)
                timesteps = get_schedule(cfg.num_steps, x.shape[1])

            if model_info["guidance_distilled"]:
                x = denoise(
                    model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=cfg.guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                    inpaint_mask=inpaint_mask_seq,
                    orig_img_seq=orig_img_seq,
                    noise_seq=noise_seq,
                )
            else:
                x = denoise_cfg(
                    model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=cfg.guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                    inpaint_mask=inpaint_mask_seq,
                    orig_img_seq=orig_img_seq,
                    noise_seq=noise_seq,
                )
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            x = ae.decode(x).float()
            # x = embed_watermark(x)

            if cpu_offloading:
                model = model.cpu()
                torch.cuda.empty_cache()
                text_encoder = text_encoder.to(torch_device)

                if "klein" in model_name:
                    mod_and_upsampling_model = mod_and_upsampling_model.to(torch_device)

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        if cfg.input_image and cfg.letterboxing and orig_width is not None and orig_height is not None:
            left = (width - orig_width) // 2
            top = (height - orig_height) // 2
            right = left + orig_width
            bottom = top + orig_height
            img = img.crop((left, top, right, bottom))

        # if mod_and_upsampling_model.test_image(img):
        #     print("Your output has been flagged. Please choose another prompt / input image combination")
        # else:
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;flux2"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        img.save(output_name, exif=exif_data, quality=95, subsampling=0)
        print(f"Saved {output_name}")

    except Exception as e:  # noqa: BLE001
        print(f"\n  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("  The model is still loaded. Please fix the error and try again.\n", file=sys.stderr)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)