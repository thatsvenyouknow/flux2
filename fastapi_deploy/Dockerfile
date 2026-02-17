# FLUX.2 FastAPI Server - GPU inference container
# Base: NVIDIA CUDA 12.6 runtime with cuDNN on Ubuntu 24.04
# Must match the host driver's max supported CUDA version (nvidia-smi shows 12.6)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml README.md LICENSE.md ./
COPY src/ src/

# Install the project (editable) with PyTorch CUDA 12.6 wheels
RUN pip install --no-cache-dir --break-system-packages \
    -e . \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Install FastAPI-specific dependencies
COPY fastapi_deploy/requirements.txt fastapi_deploy/requirements.txt
RUN pip install --no-cache-dir --break-system-packages \
    -r fastapi_deploy/requirements.txt

# Copy the API code (after install so code changes don't bust the pip cache)
COPY fastapi_deploy/ fastapi_deploy/

EXPOSE 8000

# Run with uvicorn. --workers 1 ensures a single process (one GPU worker).
ENTRYPOINT ["uvicorn", "fastapi_deploy.api:app", \
    "--host", "0.0.0.0", \
    "--port", "8000"]
