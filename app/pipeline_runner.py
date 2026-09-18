"""YuE2 pipeline glue, free of any web-framework dependency.

Thin wrappers around yue2.pipeline: load the pipeline from env, run one
generation (or just the plan) into a job directory and enumerate the resulting
artifacts, and park/unpark the weights. The GPU serialization lives in the
worker; nothing here touches the _gpu lock.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from yue2.pipeline import YuE2Pipeline

from .schemas import GenerateRequest

# The keys the pipeline actually accepts as a request; file_format and stage are
# server-side delivery concerns and must NOT be forwarded to pipe(...).
_REQUEST_KEYS = ("style", "lyrics", "tags", "cot", "seed", "cfg_scale", "abc", "id",
                 "abc_sampling", "semantic_sampling")
# plan() takes no semantic_sampling.
_PLAN_KEYS = tuple(k for k in _REQUEST_KEYS if k != "semantic_sampling")


def load_pipeline() -> YuE2Pipeline:
    """Build the pipeline from env, resolving (downloading) weights eagerly.

    from_pretrained resolves and integrity-checks the weight files; the actual
    model is built onto the GPU lazily on the first generation. run_api forces
    that load at startup so /health only flips to ok once the model is resident.
    """
    model = os.environ.get("YUE_MODEL", "m-a-p/YuE2-3B")
    vae = os.environ.get("YUE_VAE", "standard")
    backend = os.environ.get("YUE_BACKEND", "torch")
    budget = float(os.environ.get("YUE_MEMORY_BUDGET_GIB", "24"))
    quantization = os.environ.get("YUE_QUANTIZATION", "none")
    revision = os.environ.get("YUE_REVISION") or None
    # "standard"/"legacy" are pipeline-native aliases; the default HF repo is
    # m-a-p/YuE2-Vae for "standard" and m-a-p/YuE2-Vae-legacy for "legacy".
    if vae == "standard":
        vae = "m-a-p/YuE2-Vae"
    elif vae == "legacy":
        vae = "m-a-p/YuE2-Vae-legacy"
    return YuE2Pipeline.from_pretrained(
        model=model, vae=vae, device="auto", memory_budget_gib=budget,
        backend=backend, quantization=quantization, revision=revision,
    )


def preload(pipe: YuE2Pipeline) -> None:
    """Load everything the first request would otherwise load lazily.

    Two lazy loads sit between "process started" and "first job done":
      - the LM (`_load_model`), built onto the GPU, and
      - the VAE decoder (`_vae`), read from disk on the first decode.
    Force both now so readiness means truly ready and the first /v1/generate
    isn't slower than the rest. The VAE is built onto CPU exactly as decode()
    does it — decode() moves it to the GPU per call, so we don't here.
    """
    pipe._load_model()
    if pipe._vae is None:
        from yue2.modeling_vae import YuE2VAE

        pipe._vae = YuE2VAE.from_pretrained(
            pipe.vae_dir, decoder_only=True, device="cpu", local_files_only=True,
        )


def park(pipe: YuE2Pipeline) -> None:
    pipe.park()


def unpark(pipe: YuE2Pipeline) -> None:
    pipe.unpark()


def _kwargs(request: GenerateRequest, keys) -> dict:
    data = request.model_dump()
    return {k: data[k] for k in keys if data.get(k) is not None}


def run_generation(pipe: YuE2Pipeline, request: GenerateRequest, job_dir: Path) -> List[dict]:
    """Generate into ``job_dir`` and return a list of {name, path} dicts.

    stage="audio" runs the full pipeline and drops the reproducible bundle;
    stage="plan" stops after the ABC score. lyrics.txt is written by hand for
    the record — save_artifacts stows the lyrics inside request.json only.
    """
    job_dir = Path(job_dir)

    if request.stage == "plan":
        plan = pipe.plan(**_kwargs(request, _PLAN_KEYS))
        plan.save(job_dir)  # score.abc (+ token .npy + plan.json + manifest)
    else:
        result = pipe(**_kwargs(request, _REQUEST_KEYS))
        result.save_artifacts(job_dir)  # audio.flac, score.abc, *.npy, *.json
        if request.file_format == "wav":
            # save_artifacts always writes FLAC; deliver WAV when asked and drop
            # the FLAC so the audio artifact is unambiguous.
            result.save(job_dir / "audio.wav")
            (job_dir / "audio.flac").unlink(missing_ok=True)

    _write_lyrics(job_dir, request.lyrics)
    return _enumerate_artifacts(job_dir)


def _write_lyrics(job_dir: Path, lyrics: str) -> None:
    (job_dir / "lyrics.txt").write_text(lyrics, encoding="utf-8")


def _enumerate_artifacts(job_dir: Path) -> List[dict]:
    """Every file the run wrote becomes a named artifact, audio first."""
    files = sorted(p for p in job_dir.iterdir() if p.is_file())
    # Surface audio + score first so a client scanning the list hits the good
    # stuff before the reproducibility .npy dumps.
    files.sort(key=lambda p: (p.name not in ("audio.flac", "audio.wav", "score.abc"), p.name))
    return [{"name": p.name, "path": str(p)} for p in files]
