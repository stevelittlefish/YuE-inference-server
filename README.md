# YuE2 Inference Server

An HTTP inference server for **[YuE2](https://github.com/multimodal-art-projection/YuE)**, the frontier text-to-music model from M·A·P. Upstream YuE2 ships a CLI only; this fork wraps the same pipeline in a small FastAPI service so YuE2 can run as a long-lived backend behind an orchestrator.

It is built to conform to the **ASS (Audio Slop Server)** backend contract — a Go orchestrator that owns one GPU and multiplexes several dockerised audio backends onto it — but the API is a plain, self-contained JSON service that is useful on its own.

> **Just want the model, the CLI, or the research?** Everything upstream — the model card, benchmarks, generation guide, covers, editing, and the agent skill — lives in the original project: **[multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)**. This repo is only about serving it over HTTP.

## What this fork adds

Everything under `app/` plus `run_api.py` and the `Dockerfile`; the `src/yue2` pipeline is upstream, unmodified except where noted.

- **`app/server.py`** — the FastAPI app and all HTTP routes.
- **`app/worker.py`** — an in-memory job queue with a single worker thread. Generation is serialized on one worker (the GPU is a single resource), so the service runs under one uvicorn worker.
- **`app/pipeline_runner.py`** — web-framework-free glue over `yue2.pipeline`: load the pipeline, run one generation into a job directory, enumerate artifacts, and park/unpark the weights.
- **`app/schemas.py`** — the request/response models.
- **`run_api.py`** — the entrypoint: load the pipeline, **preload the model and VAE onto the device at startup** (so `/health` only reports ready once inference is truly ready), then serve.

## API

All JSON unless noted. Generation takes minutes, so it is asynchronous: `POST /v1/generate` enqueues a job and returns immediately; poll `GET /v1/jobs/{id}` until it reaches `succeeded` or `failed`.

| Method & path | Purpose |
|---|---|
| `GET /health` | Readiness probe. `{"status":"ok","parked":<bool>}`, 200 only once weights are loaded. |
| `GET /v1/info` | `{model, device, sample_rate, parked, backend, verb, cot_modes, stages, file_formats}`. |
| `POST /v1/generate` | Submit a job. Body is the generation request (below). Returns `{job_id, state}`. |
| `GET /v1/jobs` | List all jobs. |
| `GET /v1/jobs/{id}` | Job status: `{job_id, state, error, timestamps, queue_position, request, artifacts:[…]}`. `state ∈ queued\|running\|succeeded\|failed`. |
| `GET /v1/jobs/{id}/result/{name}` | Download one named artifact. |
| `DELETE /v1/jobs/{id}` | Delete a job and its files. |
| `POST /park` | Move weights to CPU RAM and free the GPU. Idempotent, no auth (orchestrator-plane). |
| `POST /unpark` | Move weights back onto the GPU. Idempotent, no auth. |

**Park / unpark** lets an orchestrator free the card for another model without a full container restart. When parked the backend is still "up" for readiness; the orchestrator unparks before sending work.

### The generation request

`POST /v1/generate` takes YuE2's parameters as JSON (no audio upload):

| Field | Meaning |
|---|---|
| `lyrics` (required) | The song lyrics, e.g. `"[Verse]\n…\n[Chorus]\n…"`. |
| `style` / `tags` | Style prompt, e.g. `"Mandarin, warm piano, acoustic pop, female vocal"`. `tags` is an older alias for `style`; supply one. |
| `cot` | Planning depth: `full` (editable melody-and-chord plan, default for new songs), `melody` (recommended for covers), or `off`. |
| `abc` | A pre-supplied ABC score, which skips the planning phase. |
| `stage` | `audio` (full song, default) or `plan` (ABC score only — fast). |
| `file_format` | `flac` (default) or `wav`. |
| `seed`, `cfg_scale`, `abc_sampling`, `semantic_sampling`, `id` | Optional sampling and identity knobs; unset ones fall back to the pipeline's defaults. |

Unknown fields are rejected with a 422. A successful job's `artifacts` enumerates every output — audio first, then the score, lyrics, and reproducibility metadata (semantic tokens, latents, settings).

```bash
# Submit
curl -s localhost:5340/v1/generate -H 'content-type: application/json' -d '{
  "style": "Mandarin, warm piano, acoustic pop, female vocal",
  "lyrics": "[Verse]\nline one\nline two\n[Chorus]\nhook",
  "cot": "full", "seed": 42
}'
# -> {"job_id":"…","state":"queued"}

# Poll, then download the audio
curl -s localhost:5340/v1/jobs/<job_id>
curl -sO localhost:5340/v1/jobs/<job_id>/result/audio.flac
```

## Running it

**Requirements:** Linux · NVIDIA GPU with BF16 support and ~24 GB VRAM. Model weights download from Hugging Face on first use.

### Docker (recommended)

The included `Dockerfile` builds a CUDA 12.8 image with a pinned GPU torch and the source on `PYTHONPATH`.

```bash
docker build -t yue2-inference-server .
docker run --gpus all -p 5340:5340 \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  yue2-inference-server
```

Weights authenticate from a token *file* (never bake a token into the image); mount it at `HF_TOKEN_PATH` (default `/cache/hf-token`) if the model requires auth.

### Local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python run_api.py
```

### Configuration (environment variables)

The service is configured entirely by environment — no config files.

| Variable | Default | Purpose |
|---|---|---|
| `YUE_MODEL` | `m-a-p/YuE2-3B` | DiT/LM weights. |
| `YUE_VAE` | `standard` | Decoder: `standard` · `legacy` · a local path · an HF repo. |
| `YUE_BACKEND` | `torch` | `torch` · `torch-eager` · `vllm`. |
| `YUE_MEMORY_BUDGET_GIB` | `24` | Per-process VRAM cap, GiB. |
| `YUE_QUANTIZATION` | `none` | `none` · `fp8`. |
| `YUE_REVISION` | — | Optional HF revision for the model weights. |
| `YUE_API_PORT` | `5340` | Listen port. |
| `YUE_OUTPUT_ROOT` | `/app/outputs/jobs` | Where job outputs are written. |
| `YUE_API_KEY` | off | Optional bearer / `X-API-Key` auth for the `/v1` routes. |
| `YUE_DEFAULT_COT` | pipeline default (`full`) | Default `cot` when a request omits it. |
| `YUE_DEFAULT_FILE_FORMAT` | `flac` | Default audio format. |

## License

Unchanged from upstream. **Code, agent skill, and documentation:** [Apache 2.0](LICENSE). **Model weights:** [CC BY-NC 4.0 with additional creator permission](MODEL_LICENSE). See the upstream project for the full model card and terms. Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
