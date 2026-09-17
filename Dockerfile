# syntax=docker/dockerfile:1

# CUDA 12.8 runtime (not devel — no build step needs nvcc). Matches the cu128
# torch wheels installed below; torch bundles its own CUDA libs, the base image
# supplies the driver-facing runtime + libsndfile/ffmpeg soundfile needs.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Generic cache defaults; ASS overrides HF_HOME/TORCH_HOME to /cache/yue/...
    # per service. HF_TOKEN_PATH is the ONE shared token file every backend
    # reads — never per-service.
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    HF_TOKEN_PATH=/cache/hf-token

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Heavy, stable dependency layers FIRST, before any source. A source edit
#     must never invalidate the multi-GB torch layer. ---

# torch 2.10.0 from the CUDA 12.8 wheel index — pinned so we never fall back to
# CPU wheels and leave the GPU idle. This is the big cached layer.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0

# The rest of the pinned deps (torch is already satisfied above).
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Tiny source layer LAST. ---
COPY . .

# Install THIS package only (deps already present), no re-resolution.
RUN pip3 install --no-cache-dir --no-deps -e .

RUN mkdir -p /app/outputs/jobs /cache/huggingface /cache/torch

# The real serving port (§7); the container listens on YUE_API_PORT (same number).
EXPOSE 5340

# Single uvicorn worker — run_api.py enforces workers=1 (serial GPU, in-proc store).
CMD ["python3", "run_api.py"]
