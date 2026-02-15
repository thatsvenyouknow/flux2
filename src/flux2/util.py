import base64
import io
import os
import sys

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import FP8Linear, Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .text_encoder import load_mistral_small_embedder, load_qwen3_embedder

import os
env_vars = os.environ
os.environ["AE_MODEL_PATH"] = env_vars.get("AE_MODEL_PATH", "/data/image_models/models/diffusers/models--black-forest-labs--FLUX.2-klein-4B/ae.safetensors")
os.environ["KLEIN_4B_MODEL_PATH"] = env_vars.get("KLEIN_4B_MODEL_PATH", "/data/image_models/models/diffusers/models--black-forest-labs--FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
os.environ["KLEIN_4B_FP8_MODEL_PATH"] = env_vars.get("KLEIN_4B_FP8_MODEL_PATH", "/data/image_models/models/diffusers/models--black-forest-labs--FLUX.2-klein-4B/flux-2-klein-4b-fp8.safetensors")
os.environ["KLEIN_4B_BASE_MODEL_PATH"] = env_vars.get("KLEIN_4B_BASE_MODEL_PATH", "/data/image_models/models/diffusers/models--black-forest-labs--FLUX.2-klein-4B/flux-2-klein-base-4b.safetensors")
os.environ["KLEIN_9B_MODEL_PATH"] = env_vars.get("KLEIN_9B_MODEL_PATH", "/data/image_models/models/diffusers/models--black-forest-labs--FLUX.2-klein-9B/flux-2-klein-9b.safetensors")
os.environ["KLEIN_9B_FP8_MODEL_PATH"] = env_vars.get("KLEIN_9B_FP8_MODEL_PATH", "/data/image_models/models/diffusers/models--black-forest-labs--FLUX.2-klein-9B/flux-2-klein-9b-fp8.safetensors")

FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda", load_in_8bit=False: load_qwen3_embedder(variant="4B", device=device, load_in_8bit=load_in_8bit),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
        "optimizations": {"quantize_text_encoder": True, "compile_model": True, "cpu_offloading": False},
    },
    "flux.2-klein-4b-fp8": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "filename": "flux-2-klein-4b-fp8.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda", load_in_8bit=False: load_qwen3_embedder(variant="4B", device=device, load_in_8bit=load_in_8bit),
        "model_path": "KLEIN_4B_FP8_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
        "optimizations": {"quantize_text_encoder": True, "compile_model": False, "cpu_offloading": False}, #compile=False bc RTX 3090 doesn't support FP8 computing
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda", load_in_8bit=False: load_qwen3_embedder(variant="4B", device=device, load_in_8bit=load_in_8bit),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
        "optimizations": {"quantize_text_encoder": True, "compile_model": True, "cpu_offloading": False},
    },
    "flux.2-klein-9b-fp8": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9b-fp8",
        "filename": "flux-2-klein-9b-fp8.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda", load_in_8bit=False: load_qwen3_embedder(variant="8B", device=device, load_in_8bit=load_in_8bit),
        "model_path": "KLEIN_9B_FP8_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
        "optimizations": {"quantize_text_encoder": True, "compile_model": False, "cpu_offloading": False}, #compile_model=False: RTX 3090 doesn't support FP8 computing
    },
    # "flux.2-klein-9b": {
    #     "repo_id": "black-forest-labs/FLUX.2-klein-9B",
    #     "filename": "flux-2-klein-9b.safetensors",
    #     "filename_ae": "ae.safetensors",
    #     "params": Klein9BParams(),
    #     "text_encoder_load_fn": lambda device="cuda", load_in_8bit=False: load_qwen3_embedder(variant="8B", device=device, load_in_8bit=load_in_8bit),
    #     "model_path": "KLEIN_9B_MODEL_PATH",
    #     "defaults": {"guidance": 1.0, "num_steps": 4},
    #     "fixed_params": {"guidance", "num_steps"},
    #     "guidance_distilled": True,
    #     "optimizations": {"quantize_text_encoder": True, "compile_model": False, "cpu_offloading": True},
    # },
    # "flux.2-klein-base-9b": {
    #     "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
    #     "filename": "flux-2-klein-base-9b.safetensors",
    #     "filename_ae": "ae.safetensors",
    #     "params": Klein9BParams(),
    #     "text_encoder_load_fn": lambda device="cuda", load_in_8bit=False: load_qwen3_embedder(variant="8B", device=device, load_in_8bit=load_in_8bit),
    #     "model_path": "KLEIN_9B_BASE_MODEL_PATH",
    #     "defaults": {"guidance": 4.0, "num_steps": 50},
    #     "fixed_params": {},
    #     "guidance_distilled": False,
    # },
    # "flux.2-dev": {
    #     "repo_id": "black-forest-labs/FLUX.2-dev",
    #     "filename": "flux2-dev.safetensors",
    #     "filename_ae": "ae.safetensors",
    #     "params": Flux2Params(),
    #     "text_encoder_load_fn": load_mistral_small_embedder,
    #     "model_path": "FLUX2_MODEL_PATH",
    #     "defaults": {"guidance": 4.0, "num_steps": 50},
    #     "fixed_params": {},
    #     "guidance_distilled": True,
    # },
}

def get_model_optimizations(model_name: str) -> dict:
    """Return optimizations for a model. compile_model is forced False when cpu_offloading (incompatible)."""
    info = FLUX2_MODEL_INFO.get(model_name.lower(), {})
    opts = dict(info.get("optimizations", {}))
    if opts.get("cpu_offloading"):
        opts["compile_model"] = False
    return opts


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        if config["model_path"] in os.environ:
            weight_path = os.environ[config["model_path"]]
            assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
        else:
            # download from huggingface
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(
                    f"Failed to access the model repository. Please check your internet "
                    f"connection and make sure you've access to {config['repo_id']}."
                    "Stopping."
                )
                sys.exit(1)

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))

        # FP8 checkpoints carry weight_scale / input_scale per layer.
        # Swap matching nn.Linear → FP8Linear so weights stay in FP8 (~1 byte/param).
        fp8_layers = {k.removesuffix(".weight_scale") for k in sd if k.endswith(".weight_scale")}
        if fp8_layers:
            print(f"Detected FP8 checkpoint – converting {len(fp8_layers)} layers to FP8Linear")
            for layer_path in fp8_layers:
                parts = layer_path.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                attr = parts[-1]
                old = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
                with torch.device("meta"):
                    fp8 = FP8Linear(old.in_features, old.out_features, bias=old.bias is not None)
                if attr.isdigit():
                    parent[int(attr)] = fp8
                else:
                    setattr(parent, attr, fp8)
            # input_scale is for FP8 activation quantization – not needed for bf16 inference
            sd = {k: v for k, v in sd.items() if not k.endswith(".input_scale")}

        model.load_state_dict(sd, strict=True, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(model_name: str, device: str | torch.device = "cuda", load_in_8bit: bool = False):
    config = FLUX2_MODEL_INFO[model_name.lower()]
    return config["text_encoder_load_fn"](device=device, load_in_8bit=load_in_8bit)


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if "AE_MODEL_PATH" in os.environ:
        weight_path = os.environ["AE_MODEL_PATH"]
        assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
    else:
        # download from huggingface
        try:
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename_ae"],
                repo_type="model",
            )
        except huggingface_hub.errors.RepositoryNotFoundError:
            print(
                f"Failed to access the model repository. Please check your internet "
                f"connection and make sure you've access to {config['repo_id']}."
                "Stopping."
            )
            sys.exit(1)

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
