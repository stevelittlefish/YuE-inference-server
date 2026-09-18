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

# Install THIS package only (deps already present), no re-resolution. Non-editable.
# NOTE: on its own this proved unreliable in the image — the build silently
# installed ZERO packages (v1.0.1 shipped without an importable `yue2`, so
# run_api.py died on `ModuleNotFoundError: No module named 'yue2'`). Rather than
# fight the system pip's src-layout/PEP-639 discovery, we ALSO put the copied
# source on PYTHONPATH below, which is exactly what the repo's own pytest config
# does (`pythonpath = ["src"]`). Belt and suspenders: whichever works, `import
# yue2` resolves. `|| true` so a discovery hiccup can't fail the build — the
# PYTHONPATH line is the real guarantee.
RUN pip3 install --no-cache-dir --no-deps . || true

# The source lives at /app/src (src-layout); put it on the path so `import yue2`
# works deterministically regardless of what the wheel build did.
ENV PYTHONPATH=/app/src

RUN mkdir -p /app/outputs/jobs /cache/huggingface /cache/torch

# The real serving port (§7); the container listens on YUE_API_PORT (same number).
EXPOSE 5340

# Single uvicorn worker — run_api.py enforces workers=1 (serial GPU, in-proc store).
CMD ["python3", "run_api.py"]
