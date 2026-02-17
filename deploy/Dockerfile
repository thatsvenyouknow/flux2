# FLUX.2 Streamlit App - GPU inference container
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

# Install the project (editable) with PyTorch CUDA 12.6 wheels + Streamlit deps
COPY scripts/requirements-app.txt scripts/requirements-app.txt
RUN pip install --no-cache-dir --break-system-packages \
    -e . \
    --extra-index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir --break-system-packages \
    -r scripts/requirements-app.txt

# Copy application scripts (after install so code changes don't bust the pip cache)
COPY scripts/ scripts/

# Streamlit configuration: disable telemetry, set server defaults
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
ENV STREAMLIT_SERVER_MAX_MESSAGE_SIZE=50
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "scripts/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]
