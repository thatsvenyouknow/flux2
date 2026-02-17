"""
Streamlit app for FLUX.2 image generation.

Run with:
    streamlit run scripts/app.py

Dependencies: pip install -r scripts/requirements-app.txt
"""

import io
import os
import threading
import tempfile
from pathlib import Path
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas
from streamlit_image_comparison import image_comparison

from flux2.pipeline import GenerationOOMError


# ── GPU memory helpers ──────────────────────────────────────────────────────
def get_gpu_memory_info(device: int = 0) -> dict:
    """Return GPU memory stats in MB.

    Keys: total, allocated, reserved, free_in_reserved, free_total.
    - *free_total* = total - allocated  (usable after empty_cache)
    - *free_in_reserved* = reserved - allocated (usable right now without new alloc)
    Returns zeros if CUDA is unavailable or the context is corrupted.
    """
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "reserved": 0, "free_in_reserved": 0, "free_total": 0}
    try:
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        return {
            "total": round(total),
            "allocated": round(allocated),
            "reserved": round(reserved),
            "free_in_reserved": round(reserved - allocated),
            "free_total": round(total - allocated),
        }
    except (torch.cuda.CudaError, RuntimeError):
        return {"total": 0, "allocated": 0, "reserved": 0, "free_in_reserved": 0, "free_total": 0}

# ── Pipeline (cached across reruns) ──────────────────────────────────────────


@st.cache_resource
def load_pipeline(model_name: str, cpu_offloading: bool):
    """Load pipeline for a model with its configured optimizations.

    cpu_offloading is user-controlled; compile_model is derived from the
    model defaults but forced False when cpu_offloading is True (incompatible).
    """
    from flux2.pipeline import Flux2Pipeline
    from flux2.util import get_model_optimizations

    opts = get_model_optimizations(model_name)
    compile_model = opts["compile_model"] and not cpu_offloading
    pipe = Flux2Pipeline(
        model_name,
        cpu_offloading=cpu_offloading,
        quantize_text_encoder=opts["quantize_text_encoder"],
        compile_model=compile_model,
    )
    pipe.warmup()
    return pipe


@st.cache_resource
def get_gpu_lock():
    """Global lock shared across all Streamlit sessions.

    Ensures only one GPU operation (generation or model switch) runs at a time.
    Streamlit runs each user session in a separate thread, so threading.Lock
    is sufficient (the GIL is released during CUDA kernel execution).
    """
    return threading.Lock()


@st.cache_resource
def get_loaded_model_tracker():
    """Global tracker for which model and settings are currently loaded.

    Unlike st.session_state this survives page refreshes and is shared across
    sessions, so we always know what load_pipeline has cached.
    """
    return {"name": None, "cpu_offloading": None}


# ── Temp-file helpers (pipeline expects paths) ──────────────────────────────


def save_uploaded_to_temp(uploaded_file, suffix=".png") -> Path:
    """Save a Streamlit UploadedFile to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return Path(tmp.name)


def save_pil_to_temp(pil_image: Image.Image, suffix=".png") -> Path:
    """Save a PIL Image to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    pil_image.save(tmp.name)
    tmp.close()
    return Path(tmp.name)


# ── Model constants (from util.FLUX2_MODEL_INFO) ──────────────────────────────

def _get_model_options():
    from flux2.util import FLUX2_MODEL_INFO
    return list(FLUX2_MODEL_INFO.keys())


def _get_defaults(model_name: str):
    from flux2.util import FLUX2_MODEL_INFO
    info = FLUX2_MODEL_INFO.get(model_name)
    return info["defaults"] if info else {"num_steps": 4, "guidance": 4.0}


def _get_optimizations(model_name: str):
    from flux2.util import get_model_optimizations
    return get_model_optimizations(model_name)


def _handle_generation_error(e: Exception):
    """Show appropriate error message for generation failures."""
    if isinstance(e, GenerationOOMError):
        st.error(
            f"**Out of GPU memory.** {e}\n\n"
            "The GPU cache has been cleared automatically. You can retry with:\n"
            "- Smaller output resolution\n"
            "- Fewer or smaller reference images\n"
            "- Click **Clear GPU cache** in the sidebar if memory stays high"
        )
    else:
        st.error(f"Generation failed: {e}")


def _fatal_cuda_error():
    """Display a fatal CUDA error with a restart button.

    When running inside Docker with ``restart: unless-stopped``, killing the
    process causes Docker to bring it back automatically.
    """
    st.error(
        "**Fatal CUDA error** — the GPU is in an unrecoverable state."
    )
    if st.button("🔄 Restart app", type="primary"):
        os._exit(1)
    st.stop()


# ── App ──────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="FLUX.2 [Klein] Playground", layout="wide")
    st.title("FLUX.2 [Klein] Playground")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        model_options = _get_model_options()
        model_name = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            key="model_select",
            help="Switch between available FLUX.2 [klein] models. Loading a new model may take a moment.",
        )

        seed = st.number_input("Seed", value=42, min_value=0, step=1)
        random_seed = st.checkbox("Random seed")
        if random_seed:
            import random

            seed = random.randint(0, 2**32 - 1)
            st.caption(f"Seed: {seed}")

        defaults = _get_defaults(model_name)
        num_steps = st.number_input(
            "Steps",
            value=defaults["num_steps"],
            min_value=1,
            max_value=100,
            step=1,
            key=f"num_steps_{model_name}",
            help=f"Recommended number of steps: {defaults['num_steps']}.",
        )
        guidance = st.number_input(
            "Guidance",
            value=defaults["guidance"],
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            format="%.1f",
            key=f"guidance_{model_name}",
            help=f"Recommended guidance strength: {defaults['guidance']}.",
        )

        st.divider()
        st.subheader("Optimizations")
        opts = _get_optimizations(model_name)
        cpu_offloading = st.checkbox(
            "CPU offloading",
            value=opts["cpu_offloading"],
            key=f"cpu_offload_{model_name}",
            help="Offload model/text-encoder to CPU when not in use. Saves VRAM but slows generation.",
        )
        effective_compile = opts["compile_model"] and not cpu_offloading
        st.caption("Active for this model:")
        st.markdown(f"{'✅' if opts['quantize_text_encoder'] else '⬜'} INT8 text encoder")
        st.markdown(f"{'✅' if effective_compile else '⬜'} torch.compile")
        if opts["compile_model"] and cpu_offloading:
            st.caption("⚠️ torch.compile disabled (incompatible with CPU offloading)")

        # ── GPU info ──────────────────────────────────────────────────────
        st.divider()
        gpu_name = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "No GPU"
        st.subheader(f"VRAM (models)")
        st.caption(gpu_name)
        vram_placeholder = st.empty()

        if st.button("Clear GPU cache", key="clear_cache"):
            import gc
            gc.collect()
            try:
                torch.cuda.empty_cache()
                st.success("GPU cache cleared. VRAM bar updates on next interaction.")
            except (torch.cuda.CudaError, RuntimeError):
                _fatal_cuda_error()

    # Load pipeline (cached per model + offloading setting). When switching
    # models or toggling CPU offloading, acquire the GPU lock first so we don't
    # destroy the pipeline while another session is generating.
    # Uses a global tracker (not session_state) so page refreshes don't lose
    # track of what's loaded — preventing double-load OOM.
    gpu_lock = get_gpu_lock()
    loaded_model = get_loaded_model_tracker()
    needs_reload = (
        loaded_model["name"] is not None
        and (loaded_model["name"] != model_name or loaded_model["cpu_offloading"] != cpu_offloading)
    )
    if needs_reload:
        with gpu_lock:
            load_pipeline.clear()
            try:
                torch.cuda.empty_cache()
            except (torch.cuda.CudaError, RuntimeError):
                _fatal_cuda_error()
    loaded_model["name"] = model_name
    loaded_model["cpu_offloading"] = cpu_offloading
    try:
        pipeline = load_pipeline(model_name, cpu_offloading)
    except (torch.cuda.CudaError, RuntimeError) as e:
        if "CUDA" in str(e) or "illegal memory access" in str(e):
            _fatal_cuda_error()
        raise

    # ── Update VRAM bar (reflects model weights, not generation) ──────────
    mem = get_gpu_memory_info()
    with vram_placeholder.container():
        if mem["total"] > 0:
            used_pct = mem["allocated"] / mem["total"]
            st.progress(used_pct, text=f"{mem['allocated']}MB / {mem['total']}MB")
            st.caption(f"Free: {mem['free_total']}MB")
        else:
            st.warning("No GPU detected")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_t2i, tab_inpaint, tab_outpaint = st.tabs(["Text to Image", "Inpainting", "Outpainting"])

    # ── TAB 1: Text to Image ────────────────────────────────────────────────
    with tab_t2i:
        prompt = st.text_area(
            "Prompt", value="A beautiful sunset over a calm ocean", height=100, key="t2i_prompt"
        )

        col_w, col_h = st.columns(2)
        with col_w:
            width = st.number_input("Width", value=1024, min_value=256, max_value=2048, step=16, key="t2i_w")
        with col_h:
            height = st.number_input("Height", value=1024, min_value=256, max_value=2048, step=16, key="t2i_h")

        # Round to multiple of 16
        width = (width // 16) * 16
        height = (height // 16) * 16

        # Conditioning images (optional)
        cond_uploads = st.file_uploader(
            "Conditioning images (optional)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="t2i_cond",
        )
        if cond_uploads:
            cols = st.columns(min(len(cond_uploads), 4))
            for i, f in enumerate(cond_uploads):
                with cols[i % 4]:
                    st.image(f, caption=f.name, width="stretch")

        if st.button("Generate", key="t2i_gen", type="primary"):
            cond_paths = [save_uploaded_to_temp(f) for f in cond_uploads] if cond_uploads else None
            status = st.status("Queued -- waiting for GPU..." if gpu_lock.locked() else "Generating...", expanded=False)
            try:
                with gpu_lock:
                    status.update(label="Generating...", state="running")
                    result = pipeline.generate(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_steps=num_steps,
                        guidance=guidance,
                        seed=seed,
                        cond_images=cond_paths,
                    )
                    st.session_state["result_t2i"] = result
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                _handle_generation_error(e)

        # Persist result across reruns
        if "result_t2i" in st.session_state:
            img = st.session_state["result_t2i"]
            img_col, dl_col = st.columns([20, 1])
            with img_col:
                st.image(img, caption=f"Generated Image ({img.width}x{img.height})", width="stretch")
            with dl_col:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button("↓", data=buf.getvalue(), file_name="generated.png", mime="image/png", key="dl_t2i", help="Download image")

    # ── TAB 2: Inpainting ───────────────────────────────────────────────────
    with tab_inpaint:
        prompt_inp = st.text_area(
            "Prompt",
            value="Make the character smile",
            height=100,
            key="inp_prompt",
        )

        inp_upload_key = st.session_state.get("inpaint_upload_key", 0)
        uploaded_file = st.file_uploader(
            "Upload input image", type=["png", "jpg", "jpeg"], key=f"inp_upload_{inp_upload_key}"
        )

        # Input can be from upload or from "Use as input" (previous result).
        # Only re-read the file when the upload actually changes (avoids
        # stream-consumption issues on reruns) and bump the canvas key so
        # st_canvas picks up the new background image.
        if uploaded_file is not None:
            file_id = (uploaded_file.name, uploaded_file.size)
            prev_file_id = st.session_state.get("_inpaint_file_id")
            if file_id != prev_file_id:
                uploaded_file.seek(0)
                input_img = Image.open(uploaded_file).convert("RGB")
                st.session_state["inpaint_input"] = input_img
                st.session_state["_inpaint_file_id"] = file_id
                st.session_state["inpaint_canvas_key"] = (
                    st.session_state.get("inpaint_canvas_key", 0) + 1
                )
            else:
                input_img = st.session_state["inpaint_input"]
        elif "inpaint_input" in st.session_state:
            input_img = st.session_state["inpaint_input"]
        else:
            input_img = None

        if input_img is not None:
            orig_w, orig_h = input_img.size
            st.caption(f"Image size: {orig_w} x {orig_h}")

            # Canvas controls
            brush_size = st.slider("Brush size", min_value=5, max_value=100, value=30, key="brush")

            # Scale image to fit canvas display (fixed width works on both desktop & mobile)
            MAX_CANVAS_W = 300
            scale = min(MAX_CANVAS_W / orig_w, 1.0)
            canvas_w = int(orig_w * scale)
            canvas_h = int(orig_h * scale)

            inp_canvas_key = st.session_state.get("inpaint_canvas_key", 0)
            st.write("Paint the area you want to regenerate (red = masked):")
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0)",
                stroke_width=brush_size,
                stroke_color="rgba(255, 0, 0, 0.5)",
                background_image=input_img,
                drawing_mode="freedraw",
                height=canvas_h,
                width=canvas_w,
                key=f"inpaint_canvas_{inp_canvas_key}",
            )

            # Reference images (optional)
            cond_uploads_inp = st.file_uploader(
                "Reference images (optional)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="inp_cond",
                help="Upload images whose content should guide the inpainted area (e.g. a logo to paint in).",
            )
            if cond_uploads_inp:
                cols = st.columns(min(len(cond_uploads_inp), 4))
                for i, f in enumerate(cond_uploads_inp):
                    with cols[i % 4]:
                        st.image(f, caption=f.name, width="stretch")

            # Generation settings
            col_s, col_l = st.columns(2)
            with col_s:
                strength = st.slider("Strength", 0.1, 1.0, 0.85, 0.05, key="inp_strength")
            with col_l:
                letterboxing = st.checkbox(
                    "Letterboxing",
                    value=False,
                    key="inp_lb",
                    help="Pad the image to a multiple of 16px instead of cropping. "
                         "Output is cropped back to the original size, so input size = output size.",
                )

            if st.button("Generate", key="inp_gen", type="primary"):
                # Extract binary mask from canvas drawing
                has_mask = (
                    canvas_result is not None
                    and canvas_result.image_data is not None
                    and canvas_result.image_data[:, :, 3].sum() > 0
                )
                if not has_mask:
                    st.warning("Please draw a mask on the image first.")
                else:
                    mask_rgba = canvas_result.image_data
                    mask_binary = (mask_rgba[:, :, 3] > 0).astype(np.uint8) * 255
                    mask_pil = Image.fromarray(mask_binary, mode="L")
                    # Scale mask back to original image dimensions
                    mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)

                    input_path = save_uploaded_to_temp(uploaded_file) if uploaded_file else save_pil_to_temp(input_img)
                    mask_path = save_pil_to_temp(mask_pil)
                    cond_paths_inp = [save_uploaded_to_temp(f) for f in cond_uploads_inp] if cond_uploads_inp else None

                    status = st.status("Queued -- waiting for GPU..." if gpu_lock.locked() else "Generating...", expanded=False)
                    try:
                        with gpu_lock:
                            status.update(label="Generating...", state="running")
                            result = pipeline.generate(
                                prompt=prompt_inp,
                                num_steps=num_steps,
                                guidance=guidance,
                                seed=seed,
                                input_image=input_path,
                                inpainting_mask=mask_path,
                                strength=strength,
                                letterboxing=letterboxing,
                                i2i_mode="inpainting",
                                cond_images=cond_paths_inp,
                            )
                            st.session_state["result_inp"] = result
                        status.update(label="Done", state="complete")
                    except Exception as e:
                        status.update(label="Failed", state="error")
                        _handle_generation_error(e)

            # Persist result across reruns
            if "result_inp" in st.session_state:
                img = st.session_state["result_inp"]
                comp_col, dl_col = st.columns([20, 1])
                with comp_col:
                    image_comparison(
                        img1=input_img,
                        img2=img,
                        label1="Original",
                        label2=f"Result ({img.width}x{img.height})",
                        starting_position=50,
                        make_responsive=True,
                    )
                with dl_col:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button("↓", data=buf.getvalue(), file_name="inpainted.png", mime="image/png", key="dl_inp", help="Download image")
                    if st.button("↻", key="inp_use_as_input", help="Use result as input for next inpainting"):
                        st.session_state["inpaint_input"] = img.copy()
                        st.session_state["inpaint_upload_key"] = inp_upload_key + 1
                        st.session_state["inpaint_canvas_key"] = inp_canvas_key + 1
                        del st.session_state["result_inp"]
                        st.rerun()
        else:
            st.info("Upload an image to get started with inpainting.")

    # ── TAB 3: Outpainting ──────────────────────────────────────────────────
    with tab_outpaint:
        prompt_out = st.text_area(
            "Prompt",
            value="Remove the green paddings that surround the image and show what is behind them",
            height=100,
            key="out_prompt",
        )

        uploaded_out = st.file_uploader(
            "Upload input image", type=["png", "jpg", "jpeg"], key="out_upload"
        )

        if uploaded_out is not None:
            file_id_out = (uploaded_out.name, uploaded_out.size)
            prev_file_id_out = st.session_state.get("_outpaint_file_id")
            if file_id_out != prev_file_id_out:
                uploaded_out.seek(0)
                input_img_out = Image.open(uploaded_out).convert("RGB")
                st.session_state["outpaint_input"] = input_img_out
                st.session_state["_outpaint_file_id"] = file_id_out
            else:
                input_img_out = st.session_state["outpaint_input"]
            orig_w, orig_h = input_img_out.size
            st.caption(f"Original size: {orig_w} x {orig_h}")

            # Target dimensions
            col_tw, col_th = st.columns(2)
            with col_tw:
                target_w = st.number_input(
                    "Target width", value=max(orig_w + 256, 1024),
                    min_value=orig_w, max_value=4096, step=16, key="out_tw"
                )
            with col_th:
                target_h = st.number_input(
                    "Target height", value=max(orig_h + 256, 1024),
                    min_value=orig_h, max_value=4096, step=16, key="out_th"
                )

            # Round to multiple of 16
            target_w = (target_w // 16) * 16
            target_h = (target_h // 16) * 16

            # Position
            position = st.selectbox(
                "Image placement",
                ["Center", "Left", "Right", "Top", "Bottom"],
                index=0,
                key="out_pos",
            )

            # Compute offset from position name
            pos_map = {
                "Center": (None, None),
                "Left": (0, None),
                "Right": (target_w - orig_w, None),
                "Top": (None, 0),
                "Bottom": (None, target_h - orig_h),
            }
            off_x, off_y = pos_map[position]

            # Fill mode and strength
            fill_mode = st.selectbox(
                "Border fill mode",
                ["Solid Color", "Reflect", "Blur", "Edge Repeat"],
                index=0,
                key="out_fill_mode",
                help="How to initialize the border area before generation. "
                     "Reflect and Blur produce more natural results than solid color.",
            )
            fill_mode_map = {
                "Solid Color": "color",
                "Reflect": "reflect",
                "Blur": "blur",
                "Edge Repeat": "edge",
            }
            fill_mode_val = fill_mode_map[fill_mode]

            fill_color = "#09F507"
            if fill_mode == "Solid Color":
                fill_color = st.color_picker("Fill color", value=fill_color, key="out_fill")

            strength_out = st.slider("Strength", 0.5, 1.0, 0.85, 0.05, key="out_strength")

            # Reference images (optional)
            cond_uploads_out = st.file_uploader(
                "Reference images (optional)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="out_cond",
                help="Upload images whose content should guide the outpainted area.",
            )
            if cond_uploads_out:
                cols = st.columns(min(len(cond_uploads_out), 4))
                for i, f in enumerate(cond_uploads_out):
                    with cols[i % 4]:
                        st.image(f, caption=f.name, width="stretch")

            # Preview: show placement on canvas with chosen fill mode
            paste_x = off_x if off_x is not None else (target_w - orig_w) // 2
            paste_y = off_y if off_y is not None else (target_h - orig_h) // 2
            pad_l, pad_t = paste_x, paste_y
            pad_r = target_w - (paste_x + orig_w)
            pad_b = target_h - (paste_y + orig_h)
            img_arr = np.array(input_img_out)

            if fill_mode_val == "color":
                preview = Image.new("RGB", (target_w, target_h), fill_color)
                preview.paste(input_img_out, (paste_x, paste_y))
            elif fill_mode_val in ("reflect", "edge"):
                np_mode = "reflect" if fill_mode_val == "reflect" else "edge"
                preview_arr = np.pad(img_arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=np_mode)
                preview = Image.fromarray(preview_arr)
            else:  # blur
                preview_arr = np.pad(img_arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
                preview = Image.fromarray(preview_arr)
                blurred = preview.filter(ImageFilter.GaussianBlur(radius=max(pad_l, pad_t, pad_r, pad_b) // 2))
                blur_blend = Image.new("L", (target_w, target_h), 255)
                blur_blend.paste(Image.new("L", (orig_w, orig_h), 0), (paste_x, paste_y))
                blur_blend = blur_blend.filter(ImageFilter.GaussianBlur(radius=32))
                preview = Image.composite(blurred, preview, blur_blend)

            st.image(preview, caption=f"Preview ({fill_mode}): {target_w} x {target_h}", width="stretch")

            if st.button("Generate", key="out_gen", type="primary"):
                input_path = save_uploaded_to_temp(uploaded_out)
                cond_paths_out = [save_uploaded_to_temp(f) for f in cond_uploads_out] if cond_uploads_out else None
                status = st.status("Queued -- waiting for GPU..." if gpu_lock.locked() else "Generating...", expanded=False)
                try:
                    with gpu_lock:
                        status.update(label="Generating...", state="running")
                        result = pipeline.generate(
                            prompt=prompt_out,
                            width=target_w,
                            height=target_h,
                            num_steps=num_steps,
                            guidance=guidance,
                            seed=seed,
                            input_image=input_path,
                            i2i_mode="outpainting",
                            strength=strength_out,
                            offset_x=off_x,
                            offset_y=off_y,
                            fill_mode=fill_mode_val,
                            fill_color=fill_color,
                            cond_images=cond_paths_out,
                        )
                        st.session_state["result_out"] = result
                    status.update(label="Done", state="complete")
                except Exception as e:
                    status.update(label="Failed", state="error")
                    _handle_generation_error(e)

            # Persist result across reruns
            if "result_out" in st.session_state:
                img = st.session_state["result_out"]
                comp_col, dl_col = st.columns([20, 1])
                with comp_col:
                    image_comparison(
                        img1=preview,
                        img2=img,
                        label1="Input placement",
                        label2=f"Result ({img.width}x{img.height})",
                        starting_position=50,
                        make_responsive=True,
                    )
                with dl_col:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button("↓", data=buf.getvalue(), file_name="outpainted.png", mime="image/png", key="dl_out", help="Download image")
        else:
            st.info("Upload an image to get started with outpainting.")


if __name__ == "__main__":
    main()