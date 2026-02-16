import gc
import random
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from einops import rearrange
from PIL import ExifTags, Image, ImageFilter

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
from flux2.util import FLUX2_MODEL_INFO, get_model_optimizations, load_ae, load_flow_model, load_text_encoder

# from flux2.watermark import embed_watermark


class GenerationOOMError(RuntimeError):
    """Raised when a generation fails due to GPU out-of-memory.

    The pipeline has already cleaned up intermediates and freed CUDA cache,
    so the caller can safely retry with smaller settings.
    """
    pass

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
class Flux2Pipeline:

    def __init__(
        self,
        model_name: str = "flux.2-klein-4b",
        debug_mode: bool = False,
        cpu_offloading: bool = False,
        quantize_text_encoder: bool = True,
        compile_model: bool = False,
    ):
        #Set up variables
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.model_info = FLUX2_MODEL_INFO[model_name]
        self.torch_device = torch.device("cuda")

        #Optimizations
        self.cpu_offloading = cpu_offloading
        self.quantize_text_encoder = quantize_text_encoder
        self.compile_model = compile_model

        # Allow TF32 for matmuls — significant speedup on Ampere+ GPUs, negligible precision loss
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # When the text encoder is quantized to INT8, it uses ~4GB instead of ~8GB,
        # so all models fit on a 24GB GPU without offloading.
        if self.quantize_text_encoder:
            print("INT8 text encoder enabled.")

        #Load models
        self.load_models(model_name)

    def load_models(self, model_name: str):
        if hasattr(self, "model") and self.model_name == model_name:
            return  #Already loaded

        first_load = not hasattr(self, "model") or self.model is None

        if first_load:
            # No model loaded yet: load all models
            print(f"Loading models for {model_name}...")
            self.model_name = model_name
            self.model_info = FLUX2_MODEL_INFO[model_name]
            self.text_encoder = load_text_encoder(model_name, device=self.torch_device, load_in_8bit=self.quantize_text_encoder)
            self.model = load_flow_model(model_name, debug_mode=self.debug_mode, device="cpu" if self.cpu_offloading else self.torch_device)
            self.ae = load_ae(model_name)
            self.text_encoder.eval()
            self.model.eval()
            self.ae.eval()

        else:
            # Unload current model and text_encoder, then load new ones (AE stays the same)
            print(f"Unloading current model and text encoder for {self.model_name}...")
            del self.model
            del self.text_encoder
            torch.cuda.empty_cache()
            print(f"Loading models for {model_name}...")
            opts = get_model_optimizations(model_name)
            self.cpu_offloading = opts["cpu_offloading"]
            self.compile_model = opts["compile_model"]
            self.quantize_text_encoder = opts["quantize_text_encoder"]
            self.model_name = model_name
            self.model_info = FLUX2_MODEL_INFO[model_name]
            self.text_encoder = load_text_encoder(model_name, device=self.torch_device, load_in_8bit=self.quantize_text_encoder)
            self.model = load_flow_model(model_name, debug_mode=self.debug_mode, device="cpu" if self.cpu_offloading else self.torch_device)
            self.text_encoder.eval()
            self.model.eval()

        # torch.compile: flow model every time (new on switch), AE decoder only on first load (shared, already compiled otherwise)
        if self.compile_model:
            print("Compiling flow model with torch.compile...")
            self.model = torch.compile(self.model)
            if first_load:
                print("Compiling AE decoder with torch.compile...")
                self.ae.decoder = torch.compile(self.ae.decoder)
            self.warmup(height=256, width=256, num_steps=1)

    def warmup(self, height: int = 256, width: int = 256, num_steps: int = 1):
        """
        Run a tiny dummy generation to trigger torch.compile tracing.
        Call once after init so the user's first real generation is fast.
        Note:No-op when compile_model is False (e.g. FP8 models).
        """
        if not self.compile_model:
            return
        print(f"Warmup: running {num_steps}-step generation at {width}x{height} to trigger compilation...")
        with torch.inference_mode():
            #Encode a short dummy prompt
            if self.model_info["guidance_distilled"]:
                ctx = self.text_encoder(["warmup"]).to(torch.bfloat16)
            else:
                ctx_empty = self.text_encoder([""]).to(torch.bfloat16)
                ctx_prompt = self.text_encoder(["warmup"]).to(torch.bfloat16)
                ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
            ctx, ctx_ids = batched_prc_txt(ctx)

            # Generate random noise
            shape = (1, 128, height // 16, width // 16)
            randn = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
            x, x_ids = batched_prc_img(randn)
            timesteps = get_schedule(num_steps, x.shape[1])

            #Run denoise (triggers flow model compilation on first call)
            x = self._denoise(
                x=x, x_ids=x_ids, ctx=ctx, ctx_ids=ctx_ids,
                timesteps=timesteps, guidance=1.0,
                img_cond_seq=None, img_cond_seq_ids=None,
                inpaint_mask_seq=None, orig_img_seq=None, noise_seq=None,
            )

            #Run AE decode (triggers decoder compilation on first call)
            self.ae.decode(x)

            #Free intermediates
            del ctx, ctx_ids, randn, x, x_ids, timesteps
            torch.cuda.empty_cache()
        print("Warmup complete.")

    def clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (torch.cuda.CudaError, RuntimeError):
                pass

    def generate(
        self,
        #T2I
        prompt: str,
        height: int = 1360,
        width: int = 768,
        num_steps: int = 4,
        guidance: int = 4.0,
        seed: int = None,
        cond_images: List[Path] = None,
        match_image_size: Optional[int] = None,
        upsample_prompt_mode: Literal["none", "local"] = "none",
        #I2I-Inpainting
        input_image: Optional[Path] = None,
        inpainting_mask: Optional[Path] = None,
        strength: float = 0.85,
        letterboxing: bool = False,
        letterboxing_color: str = "#000000",
        i2i_mode: Literal["none", "inpainting", "outpainting"] = "none",
        #I2I-Outpainting
        offset_x: int | None = None,    # Pixel offset from left. None = center.
        offset_y: int | None = None,    # Pixel offset from top. None = center.
        fill_mode: Literal["color", "reflect", "edge", "blur"] = "reflect",
        fill_color: str = "#808080",    # Only used when fill_mode="color"
        ) -> Image.Image:

        try:
            return self._generate_inner(
                prompt=prompt, height=height, width=width,
                num_steps=num_steps, guidance=guidance, seed=seed,
                cond_images=cond_images, match_image_size=match_image_size,
                upsample_prompt_mode=upsample_prompt_mode,
                input_image=input_image, inpainting_mask=inpainting_mask,
                strength=strength, letterboxing=letterboxing,
                letterboxing_color=letterboxing_color, i2i_mode=i2i_mode,
                offset_x=offset_x, offset_y=offset_y,
                fill_mode=fill_mode, fill_color=fill_color,
            )
        except torch.cuda.OutOfMemoryError:
            #Clean up and give caller a chance to retry
            self.clear_cache()
            raise GenerationOOMError(
                f"CUDA out of memory during generation at {width}x{height} "
                f"({width * height:,} output pixels). "
                f"Try reducing the output resolution or reference image sizes."
            )

    def _generate_inner(
        self,
        prompt: str,
        height: int,
        width: int,
        num_steps: int,
        guidance: float,
        seed: int | None,
        cond_images: List[Path] | None,
        match_image_size: int | None,
        upsample_prompt_mode: str,
        input_image,
        inpainting_mask,
        strength: float,
        letterboxing: bool,
        letterboxing_color: str,
        i2i_mode: str,
        offset_x: int | None,
        offset_y: int | None,
        fill_mode: str,
        fill_color: str,
    ) -> Image.Image:
        """Inner generation logic, separated so generate() can wrap it with OOM handling."""

        # Load explicit conditioning images first
        cond_images = cond_images or []
        img_ctx = [Image.open(cond_image) for cond_image in cond_images]

        # Apply match_image_size if specified
        if match_image_size is not None:
            if match_image_size < 0 or match_image_size >= len(img_ctx):
                print(
                    f"  ! match_image_size={match_image_size} is out of range (0-{len(img_ctx)-1})",
                    file=sys.stderr,
                )
                print(f"  ! Using default dimensions: {width}x{height}", file=sys.stderr)
            else:
                ref_img = img_ctx[match_image_size]
                width, height = ref_img.size
                print(f"  Matched dimensions from image {match_image_size}: {width}x{height}")

        with torch.inference_mode():
            #Encode reference images if provided
            ref_tokens, ref_ids = encode_image_refs(self.ae, img_ctx) if img_ctx else (None, None)

            #Upsample prompt if local mode is selected
            if upsample_prompt_mode == "local":
                upsampled_prompts = self.text_encoder.upsample_prompt(
                    [prompt], img=[img_ctx] if img_ctx else None
                )
                prompt = upsampled_prompts[0] if upsampled_prompts else prompt
            else:
                prompt = prompt

            print("Generating with prompt: ", prompt)

            if self.model_info["guidance_distilled"]:
                ctx = self.text_encoder([prompt]).to(torch.bfloat16)
            else:
                ctx_empty = self.text_encoder([""]).to(torch.bfloat16)
                ctx_prompt = self.text_encoder([prompt]).to(torch.bfloat16)
                ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
            ctx, ctx_ids = batched_prc_txt(ctx)

            if self.cpu_offloading:
                self.text_encoder.cpu()
                torch.cuda.empty_cache()
                self.model.to(self.torch_device)

            orig_width = None
            orig_height = None

            if i2i_mode == "outpainting" and input_image:
                x, orig_width, orig_height, width, height = self.outpainting(
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    ref_tokens=ref_tokens,
                    ref_ids=ref_ids,
                    input_image=input_image,
                    target_width=width,
                    target_height=height,
                    strength=strength,
                    seed=seed,
                    num_steps=num_steps,
                    guidance=guidance,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    fill_mode=fill_mode,
                    fill_color=fill_color,
                )
            elif input_image:
                x, orig_width, orig_height, width, height = self.inpainting(
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    ref_tokens=ref_tokens,
                    ref_ids=ref_ids,
                    input_image=input_image,
                    inpainting_mask=inpainting_mask,
                    strength=strength,
                    letterboxing=letterboxing,
                    letterboxing_color=letterboxing_color,
                    seed=seed,
                    num_steps=num_steps,
                    guidance=guidance,
                )
            else:
                x = self.text_to_image(
                    height=height,
                    width=width,
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    ref_tokens=ref_tokens,
                    ref_ids=ref_ids,
                    seed=seed,
                    num_steps=num_steps,
                    guidance=guidance,
                )

            x = self.ae.decode(x).float()
            # x = embed_watermark(x)

            if self.cpu_offloading:
                self.model.cpu()
                torch.cuda.empty_cache()
                self.text_encoder.to(self.torch_device)


        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        if i2i_mode != "outpainting" and input_image and letterboxing and orig_width is not None and orig_height is not None:
            left = (width - orig_width) // 2
            top = (height - orig_height) // 2
            right = left + orig_width
            bottom = top + orig_height
            img = img.crop((left, top, right, bottom))

        return img
    
    def text_to_image(
        self,
        height: int,
        width: int,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        ref_tokens: torch.Tensor,
        ref_ids: torch.Tensor,
        seed: int = None,
        num_steps: int = 4,
        guidance: float = 4.0
    ) -> torch.Tensor:

        #Generate random noise in latent space
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed if seed is not None else random.randint(0, 1000000))
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")

        #Convert noise to sequence format
        x, x_ids = batched_prc_img(randn)
        timesteps = get_schedule(num_steps, x.shape[1])

        #Generate image
        x = self._denoise(
            x=x,
            x_ids=x_ids,
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
            inpaint_mask_seq=None,
            orig_img_seq=None,
            noise_seq=None,
        )

        return x

    def inpainting(
        self, 
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        ref_tokens: torch.Tensor,
        ref_ids: torch.Tensor,
        input_image: Path | Image.Image, 
        inpainting_mask: Path | Image.Image | None = None, 
        strength: float = 0.85, 
        letterboxing: bool = False, 
        letterboxing_color: str = "#000000", 
        seed: int = None, 
        num_steps: int = 4, 
        guidance: float = 4.0
        ) -> tuple[torch.Tensor, int, int, int, int]:

        #I2I placeholder
        inpaint_mask_seq = None
        orig_img_seq = None
        noise_seq = None

        #Load input image (accept Path or PIL Image)
        if isinstance(input_image, Image.Image):
            input_img = input_image.convert("RGB")
        else:
            input_img = Image.open(input_image).convert("RGB")
        orig_width, orig_height = input_img.size

        #Load inpainting mask (accept Path, PIL Image, or None)
        mask_pil = None
        if inpainting_mask is not None:
            if isinstance(inpainting_mask, Image.Image):
                mask_pil = inpainting_mask.convert("L")
            else:
                mask_pil = Image.open(inpainting_mask).convert("L")
            if mask_pil.size != input_img.size:
                mask_pil = mask_pil.resize(input_img.size, Image.NEAREST)

        #Letterbox both image and mask together (Optional)
        if letterboxing:
            letterboxed_img = letterbox_to_multiple_of_x(
                input_img, 16, color=letterboxing_color
            )
            if mask_pil is not None:
                #Letterbox mask: pad with 0 (keep original) in border areas
                lb_w, lb_h = letterboxed_img.size
                mask_letterboxed = Image.new("L", (lb_w, lb_h), 0)
                left = (lb_w - input_img.width) // 2
                top = (lb_h - input_img.height) // 2
                mask_letterboxed.paste(mask_pil, (left, top))
                mask_pil = mask_letterboxed
            input_img = letterboxed_img

        #Preprocess and encode clean input image 
        input_tensor = default_prep(input_img, limit_pixels=None, ensure_multiple=16)
        x_img = self.ae.encode(input_tensor[None].to(self.torch_device))[0].unsqueeze(0).to(torch.bfloat16)

        #Use actual latent-backed output size.
        _, _, latent_h, latent_w = x_img.shape
        width = latent_w * 16
        height = latent_h * 16

        #Generate noise in latent space
        generator = torch.Generator(device="cuda").manual_seed(seed if seed is not None else random.randint(0, 1000000))
        noise_latent = torch.randn(x_img.shape, generator=generator, dtype=torch.bfloat16, device="cuda")

        #Convert clean latents and noise to sequence format
        x_clean_seq, x_ids = batched_prc_img(x_img)
        noise_seq_full, _ = batched_prc_img(noise_latent)

        #Clean up some variables
        del x_img, noise_latent, input_tensor

        #Compute full timestep schedule, then truncate based on strength
        full_timesteps = get_schedule(num_steps, x_clean_seq.shape[1])
        num_i2i_steps = max(1, int(num_steps * strength))
        timesteps = full_timesteps[-(num_i2i_steps + 1):]
        t_start = timesteps[0]
        print(f"  img2img: strength={strength}, steps={num_i2i_steps}/{num_steps}, t_start={t_start:.4f}")

        #Create noisy latents at t_start: x_t = (1-t)*x_0 + t*noise
        x = (1 - t_start) * x_clean_seq + t_start * noise_seq_full

        #Prepare inpainting mask in latent sequence format
        if mask_pil is not None:
            #Apply same center crop as default_prep to keep mask aligned
            mask_pil = center_crop_to_multiple_of_x(mask_pil, 16)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0  # Normalize to [0, 1], preserve soft values
            mask_tensor = torch.from_numpy(mask_np).float()

            #Downsample mask to latent resolution using bilinear for smooth transitions
            mask_latent = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            ).clamp(0, 1).to(torch.bfloat16).to(self.torch_device)

            #Flatten to sequence format: (1, latent_h*latent_w, 1)
            inpaint_mask_seq = rearrange(mask_latent[0], "c h w -> (h w) c").unsqueeze(0)
            orig_img_seq = x_clean_seq
            noise_seq = noise_seq_full

            soft_px = (mask_np > 0).sum()
            hard_px = (mask_np >= 1.0).sum()
            print(f"  inpainting mask: {soft_px:.0f} px touched, {hard_px:.0f} px fully masked "
                  f"(soft={soft_px - hard_px:.0f} transition px)")

        #Generate image
        x = self._denoise(
            x=x,
            x_ids=x_ids,
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
            inpaint_mask_seq=inpaint_mask_seq,
            orig_img_seq=orig_img_seq,
            noise_seq=noise_seq,
        )

        return x, orig_width, orig_height, width, height

    def outpainting(
        self,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        ref_tokens: torch.Tensor,
        ref_ids: torch.Tensor,
        input_image: Path | Image.Image,
        target_width: int,
        target_height: int,
        strength: float = 1.0,
        seed: int = None,
        num_steps: int = 4,
        guidance: float = 4.0,
        offset_x: int | None = None,   # Pixel offset from left. None = center horizontally.
        offset_y: int | None = None,   # Pixel offset from top. None = center vertically.
        fill_mode: Literal["color", "reflect", "edge", "blur"] = "color",
        fill_color: str = "#09F507",    # Only used when fill_mode="color"
        overlap_px: int = 32,           # Pixels of the original to regenerate (overlap zone)
        feather_px: int = 40,           # Pixels of soft Gaussian transition on top of overlap
    ) -> tuple[torch.Tensor, int, int, int, int]:
        """
        Outpainting: extend an image beyond its original borders.

        Builds a canvas at (target_width, target_height), pastes the original
        image at (offset_x, offset_y), auto-generates a feathered mask with
        overlap, then delegates to the inpainting pipeline.

        The mask has three zones:
          1. Keep zone (mask=0): interior of original image, fully preserved.
          2. Overlap zone: a strip of the original that gets regenerated by the
             model. Because the model sees the original content on the canvas,
             it produces visually similar content here, so the eventual seam
             between "regenerated original" and "real original" is nearly invisible.
          3. Generate zone (mask=1): the new border area, fully generated.

        A Gaussian blur feathers the boundary between zones 1 and 2, creating a
        smooth gradient so the per-step mask blending produces a gradual transition.

        offset_x/offset_y let a UI pass exact placement (e.g. from drag-and-drop).
        When None, the image is centered on that axis.
        """

        #Load input image
        if isinstance(input_image, Image.Image):
            input_img = input_image.convert("RGB")
        else:
            input_img = Image.open(input_image).convert("RGB")
        orig_w, orig_h = input_img.size

        # Ensure target is at least as large as the original
        target_width = max(target_width, orig_w)
        target_height = max(target_height, orig_h)

        #Compute placement (default: centered)
        if offset_x is None:
            offset_x = (target_width - orig_w) // 2
        if offset_y is None:
            offset_y = (target_height - orig_h) // 2

        #Clamp offsets so the image stays within the canvas
        offset_x = max(0, min(offset_x, target_width - orig_w))
        offset_y = max(0, min(offset_y, target_height - orig_h))

        #Padding amounts on each side
        pad_l = offset_x
        pad_t = offset_y
        pad_r = target_width - (offset_x + orig_w)
        pad_b = target_height - (offset_y + orig_h)

        print(f"  outpainting: {orig_w}x{orig_h} -> {target_width}x{target_height}, "
              f"offset=({offset_x}, {offset_y}), fill_mode={fill_mode}, strength={strength}")

        #Build canvas using the chosen fill mode
        if fill_mode == "color":
            #Solid color fill 
            canvas = Image.new("RGB", (target_width, target_height), fill_color)
            canvas.paste(input_img, (offset_x, offset_y))
        elif fill_mode == "reflect":
            #Mirror/reflect the image at edges — natural continuation of patterns
            img_arr = np.array(input_img)
            canvas_arr = np.pad(
                img_arr,
                ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                mode="reflect",
            )
            canvas = Image.fromarray(canvas_arr)
        elif fill_mode == "edge":
            #Repeat edge pixels outward — simple, avoids flat color
            img_arr = np.array(input_img)
            canvas_arr = np.pad(
                img_arr,
                ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                mode="edge",
            )
            canvas = Image.fromarray(canvas_arr)
        elif fill_mode == "blur":
            #Edge-repeat then apply graduated blur that increases toward canvas edges.
            img_arr = np.array(input_img)
            canvas_arr = np.pad(
                img_arr,
                ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                mode="edge",
            )
            canvas = Image.fromarray(canvas_arr)
            #Apply heavy blur only to the border region
            blurred = canvas.filter(ImageFilter.GaussianBlur(radius=max(pad_l, pad_t, pad_r, pad_b) // 2))
            #Create a gradient mask: 0 inside original (keep sharp), 255 in border (use blurred)
            blur_blend = Image.new("L", (target_width, target_height), 255)
            blur_blend.paste(Image.new("L", (orig_w, orig_h), 0), (offset_x, offset_y))
            blur_blend = blur_blend.filter(ImageFilter.GaussianBlur(radius=max(32, feather_px)))
            canvas = Image.composite(blurred, canvas, blur_blend)
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode!r}. Use 'color', 'reflect', 'edge', or 'blur'.")

        #Auto-generate mask with overlap zone.
        #Start with everything white (255 = regenerate).
        mask = Image.new("L", (target_width, target_height), 255)

        #Compute overlap per edge: only overlap on edges that actually have a border.
        #E.g. if the image is flush against the left canvas edge, no left overlap.
        border_l = offset_x                                  # border pixels to the left
        border_t = offset_y                                  # border pixels above
        border_r = target_width - (offset_x + orig_w)       # border pixels to the right
        border_b = target_height - (offset_y + orig_h)      # border pixels below

        ol = min(overlap_px, orig_w // 4) if border_l > 0 else 0
        ot = min(overlap_px, orig_h // 4) if border_t > 0 else 0
        or_ = min(overlap_px, orig_w // 4) if border_r > 0 else 0
        ob = min(overlap_px, orig_h // 4) if border_b > 0 else 0

        #The keep region is the original image shrunk by the overlap on each edge
        keep_l = offset_x + ol
        keep_t = offset_y + ot
        keep_w = orig_w - ol - or_
        keep_h = orig_h - ot - ob

        if keep_w > 0 and keep_h > 0:
            mask.paste(Image.new("L", (keep_w, keep_h), 0), (keep_l, keep_t))

        #Feather: Gaussian blur smooths the hard edge between keep and regenerate.
        #With overlap, the gradient falls within the original image content, so
        #the model blends "regenerated original" into "real original" 
        if feather_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))

        print(f"  mask: overlap=({ol},{ot},{or_},{ob})px, feather={feather_px}px, "
              f"keep={keep_w}x{keep_h}")

        #Condition the model on the original image and the canvas mask 
        #mask_rgb = mask.convert("RGB")  #grayscale mask → 3-channel for encoder
        extra_refs = [input_img] #, mask_rgb
        extra_ref_tokens, extra_ref_ids = encode_image_refs(self.ae, extra_refs)

        #Merge with any existing conditioning references
        if ref_tokens is not None and extra_ref_tokens is not None:
            ref_tokens = torch.cat([ref_tokens, extra_ref_tokens], dim=1)
            ref_ids = torch.cat([ref_ids, extra_ref_ids], dim=1)
        elif extra_ref_tokens is not None:
            ref_tokens = extra_ref_tokens
            ref_ids = extra_ref_ids

        #Delegate to inpainting pipeline
        return self.inpainting(
            ctx=ctx,
            ctx_ids=ctx_ids,
            ref_tokens=ref_tokens,
            ref_ids=ref_ids,
            input_image=canvas,
            inpainting_mask=mask,
            strength=strength,
            letterboxing=True,  # ensure multiple-of-16 if target dims aren't
            letterboxing_color=fill_color if fill_mode == "color" else "#000000",
            seed=seed,
            num_steps=num_steps,
            guidance=guidance,
        )

    def _denoise(
        self,
        x: torch.Tensor,
        x_ids: torch.Tensor,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        timesteps: list[float],
        guidance: float,
        img_cond_seq: torch.Tensor,
        img_cond_seq_ids: torch.Tensor,
        inpaint_mask_seq: torch.Tensor | None = None,
        orig_img_seq: torch.Tensor | None = None,
        noise_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.model_info["guidance_distilled"]:
            x = denoise(
                self.model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
                inpaint_mask=inpaint_mask_seq,
                orig_img_seq=orig_img_seq,
                noise_seq=noise_seq,
            )
        else:
            x = denoise_cfg(
                self.model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
                inpaint_mask=inpaint_mask_seq,
                orig_img_seq=orig_img_seq,
                noise_seq=noise_seq,
            )
        
        return torch.cat(scatter_ids(x, x_ids)).squeeze(2)


if __name__ == "__main__":
    from pathlib import Path as _Path

    pipeline = Flux2Pipeline(cpu_offloading=False, quantize_text_encoder=True, compile_model=True)
    import time

    start = time.time()
    img = pipeline.generate(
        prompt="Picture of a sunset over the ocean.",
        height=1024,
        width=1024,
        num_steps=4,
        guidance=4.0,
        seed=42,
    )
    print(f"First generation took {time.time() - start:.2f} seconds")

    start = time.time()
    img = pipeline.generate(
        prompt="Picture of a sunset over the ocean.",
        height=1024,
        width=1024,
        num_steps=4,
        guidance=4.0,
        seed=42,
    )
    print(f"Second generation took {time.time() - start:.2f} seconds")
    # img = pipeline.generate(
    #     prompt="Make the character smile with a big happy grin, anime style",
    #     height=1024,
    #     width=1024,
    #     num_steps=4,
    #     guidance=7.5,
    #     seed=42,
    #     input_image=_Path("/home/sanctuary/projects/flux2/input_images/luffy_non16.jpg"),
    #     inpainting_mask=_Path("/home/sanctuary/projects/flux2/input_images/luffy_mouth_mask_small_non16.png"),
    #     strength=0.85,
    #     letterboxing=True,
    #     letterboxing_color="#000000",
    #     i2i_mode="inpainting",
    # )
    output_dir = _Path("output")
    output_dir.mkdir(exist_ok=True)
    output_name = output_dir / f"sample_{len(list(output_dir.glob('*')))}.png"
    img.save(output_name, quality=95, subsampling=0)
    print(f"Saved {output_name}")