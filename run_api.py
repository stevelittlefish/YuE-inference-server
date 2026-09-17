"""Entrypoint for the YuE2 music-generation service.

Loads the pipeline once (downloading weights on a cold cache), forces the model
onto the GPU so /health only reports ready when it truly is, then serves the API
with a single uvicorn worker (in-memory job store, serial GPU use).

Environment:
  YUE_MODEL             DiT/LM weights (default: m-a-p/YuE2-3B).
  YUE_VAE               VAE: standard | legacy | local path | HF repo
                        (default: standard -> m-a-p/YuE2-Vae).
  YUE_BACKEND           torch | torch-eager | vllm (default: torch — vllm pins
                        VRAM for its KV cache and fights ASS's park/share model).
  YUE_MEMORY_BUDGET_GIB Per-process VRAM cap in GiB (default: 24).
  YUE_QUANTIZATION      none | fp8 (default: none).
  YUE_REVISION          Optional HF revision for the model weights.
  YUE_API_PORT          Listen port (default: 5340). ASS injects this, not PORT.
  YUE_OUTPUT_ROOT       Directory for job outputs (default: /app/outputs/jobs).
  YUE_API_KEY           Optional bearer/X-API-Key auth (default: off).
  YUE_DEFAULT_COT       Default chain-of-thought depth when a request omits it
                        (default: unset -> the pipeline's own default, "full").
  YUE_DEFAULT_FILE_FORMAT  Default audio format (default: flac).
"""

import os

import torch
import uvicorn

from app.pipeline_runner import load_pipeline, preload
from app.server import create_app


def main():
    model_name = os.environ.get("YUE_MODEL", "m-a-p/YuE2-3B")
    backend = os.environ.get("YUE_BACKEND", "torch")
    port = int(os.environ.get("YUE_API_PORT", "5340"))
    output_root = os.environ.get("YUE_OUTPUT_ROOT", "/app/outputs/jobs")
    api_key = os.environ.get("YUE_API_KEY") or None
    default_cot = os.environ.get("YUE_DEFAULT_COT") or None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[startup] HF_HOME={os.environ.get('HF_HOME')} TORCH_HOME={os.environ.get('TORCH_HOME')}", flush=True)
    print("[startup] downloading weights (several GB, first run only) — not hung, grab a coffee", flush=True)
    pipe = load_pipeline()  # resolves + integrity-checks the weight files
    print("[startup] weights ready", flush=True)

    print(f"[startup] loading model onto {device} ...", flush=True)
    preload(pipe)  # build the model on the GPU now, so readiness means ready
    print("[startup] model resident; serving", flush=True)

    app = create_app(
        pipe=pipe,
        model_name=model_name,
        device=device,
        output_root=output_root,
        sample_rate=48000,
        backend=backend,
        api_key=api_key,
        default_cot=default_cot,
    )
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
