"""
Streamlit app for FLUX.2 image generation.

Run with:
    streamlit run scripts/app.py
"""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ── Pipeline (cached across reruns) ──────────────────────────────────────────


@st.cache_resource
def load_pipeline(model_name: str):
    from flux2.pipeline import Flux2Pipeline

    return Flux2Pipeline(model_name, cpu_offloading=True)


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


# ── Model constants ──────────────────────────────────────────────────────────

MODEL_DEFAULTS = {
    "flux.2-klein-4b": {"num_steps": 4, "guidance": 1.0},
}
MODEL_NAME = "flux.2-klein-4b"


# ── App ──────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="FLUX.2 Generator", layout="wide")
    st.title("FLUX.2 Image Generator")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        seed = st.number_input("Seed", value=42, min_value=0, step=1)
        random_seed = st.checkbox("Random seed")
        if random_seed:
            import random

            seed = random.randint(0, 2**32 - 1)
            st.caption(f"Seed: {seed}")

        st.divider()
        st.caption(
            f"Model: **{MODEL_NAME}** — "
            f"steps: {MODEL_DEFAULTS[MODEL_NAME]['num_steps']}, "
            f"guidance: {MODEL_DEFAULTS[MODEL_NAME]['guidance']} (fixed)"
        )

    # Load pipeline (cached)
    pipeline = load_pipeline(MODEL_NAME)
    defaults = MODEL_DEFAULTS[MODEL_NAME]

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_t2i, tab_inpaint = st.tabs(["Text to Image", "Inpainting"])

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
            with st.spinner("Generating..."):
                try:
                    result = pipeline.generate(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_steps=defaults["num_steps"],
                        guidance=defaults["guidance"],
                        seed=seed,
                        cond_images=cond_paths,
                    )
                    st.session_state["result_t2i"] = result
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        # Persist result across reruns
        if "result_t2i" in st.session_state:
            st.image(st.session_state["result_t2i"], caption="Generated Image", width="stretch")

    # ── TAB 2: Inpainting ───────────────────────────────────────────────────
    with tab_inpaint:
        prompt_inp = st.text_area(
            "Prompt",
            value="Make the character smile",
            height=100,
            key="inp_prompt",
        )

        uploaded_file = st.file_uploader(
            "Upload input image", type=["png", "jpg", "jpeg"], key="inp_upload"
        )

        if uploaded_file is not None:
            input_img = Image.open(uploaded_file).convert("RGB")
            orig_w, orig_h = input_img.size
            st.caption(f"Image size: {orig_w} x {orig_h}")

            # Canvas controls
            brush_size = st.slider("Brush size", min_value=5, max_value=100, value=30, key="brush")

            # Scale image to fit canvas display
            MAX_CANVAS_W = 700
            scale = min(MAX_CANVAS_W / orig_w, 1.0)
            canvas_w = int(orig_w * scale)
            canvas_h = int(orig_h * scale)

            st.write("Paint the area you want to regenerate (red = masked):")
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0)",
                stroke_width=brush_size,
                stroke_color="rgba(255, 0, 0, 0.5)",
                background_image=input_img,
                drawing_mode="freedraw",
                height=canvas_h,
                width=canvas_w,
                key="inpaint_canvas",
            )

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

                    input_path = save_uploaded_to_temp(uploaded_file)
                    mask_path = save_pil_to_temp(mask_pil)

                    with st.spinner("Generating..."):
                        try:
                            result = pipeline.generate(
                                prompt=prompt_inp,
                                num_steps=defaults["num_steps"],
                                guidance=defaults["guidance"],
                                seed=seed,
                                input_image=input_path,
                                inpainting_mask=mask_path,
                                strength=strength,
                                letterboxing=letterboxing,
                                i2i_mode="inpainting",
                            )
                            st.session_state["result_inp"] = result
                        except Exception as e:
                            st.error(f"Generation failed: {e}")

            # Persist result across reruns
            if "result_inp" in st.session_state:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(input_img, caption="Original", width="stretch")
                with col_b:
                    st.image(st.session_state["result_inp"], caption="Result", width="stretch")
        else:
            st.info("Upload an image to get started with inpainting.")


if __name__ == "__main__":
    main()
