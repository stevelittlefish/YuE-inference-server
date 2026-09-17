# BUILD SPEC — YuE2 ASS inference API server

A self-contained work order for a **fresh Claude Code session** to build the HTTP
inference server that makes this YuE2 fork an ASS (Audio Slop Server) backend. You
have no prior context; everything you need is here or at the file paths cited.

> **Read this first, then the two reference files in full:**
> `~/git/stem-separator/app/server.py` and `~/git/stem-separator/app/worker.py`.
> This server is a near-copy of that one with YuE2's generation swapped in. Don't
> reinvent the shape — mirror the reference, change the model call and the artifacts.

---

## 0. What ASS is (why this server exists)

ASS is a Go orchestrator that owns one GPU and multiplexes several dockerised
audio backends onto it — Ollama, but the "models" are whole containers. Only one
backend is resident in VRAM at a time; ASS starts a backend, waits for `/health`,
forwards a job, harvests the results over HTTP, then evicts it (stops or "parks"
it) to free the card for the next backend.

**Every backend conforms to ONE contract** so ASS speaks to all of them the same
way. We fork every upstream service to make it conform — this repo is our YuE2
fork. Your job: add the API server that implements the contract below. Upstream
YuE2 ships only a CLI (`yue2 = "yue2.cli:main"`); there is no server yet.

ASS repo (for reference, not edited by you except the `ass.toml` snippet in §7):
`~/git/AudioSlopServer` — see its `CLAUDE.md` for the full contract and house rules.

### House rules you must follow (from ASS `CLAUDE.md`)
- **Commit straight to `main` and `git push`.** No branches, no PRs. Announce
  "Slopping it straight to main!" when you commit.
- **Sarcasm in commit messages and comments.** Dry humor over dry docs.
- **No OOP ceremony** — plain functions over plain data. (FastAPI + a job manager
  is fine; don't build class hierarchies.)
- **Config via env for the container** (ASS injects env per service), no surprise
  config files. TOML is ASS's job, not the backend's.
- **Never commit an HF token.** Weights authenticate from a token file mounted
  into the container (see §6).
- Banned words: **"load-bearing"**, **"seam"**. Don't use them anywhere.

---

## 1. The backend contract (implement exactly this)

All JSON unless noted. `verb` for YuE is **`generate`**.

| Method & path | Purpose |
|---|---|
| `GET /health` | Readiness probe. Return `{"status":"ok","parked":<bool>}`. Must return 200 **only after weights are loaded** (see §4 startup). A parked backend is still "ok". |
| `GET /v1/info` | `{model, device, sample_rate, parked, ...capabilities}`. |
| `POST /v1/generate` | Submit a job. Returns `{job_id, state}`. Body = the generation request (§3). |
| `GET /v1/jobs` | List all jobs (array of job status). |
| `GET /v1/jobs/{id}` | `{job_id, state, error, created_at, started_at, finished_at, queue_position, request, artifacts:[...]}`. `state ∈ queued|running|succeeded|failed`. On success, `artifacts` enumerates every output (§5). |
| `GET /v1/jobs/{id}/result/{name}` | Download one named artifact (FileResponse). |
| `DELETE /v1/jobs/{id}` | Delete a job + its files. |
| `POST /park` | Move weights to CPU RAM, `torch.cuda.empty_cache()`, free VRAM. Idempotent. No auth (orchestrator-plane, like `/health`). |
| `POST /unpark` | Move weights back onto the GPU. Idempotent. No auth. |

**Async everywhere.** Generation takes minutes. `POST /v1/generate` enqueues and
returns immediately; the caller polls `GET /v1/jobs/{id}`. One in-memory job store,
**a single worker thread**, **`uvicorn --workers 1`** — the GPU is used serially
and the store is in-process, so multiple workers would corrupt both.

Optional bearer/`X-API-Key` auth exactly like the reference (`require_key`
dependency, off unless `YUE_API_KEY` is set). ASS reaches the backend on a private
path and sets no key, so this stays off in practice — include it for parity.

---

## 2. Reference implementation to copy (READ THESE FILES)

The stem-separator backend already implements this contract cleanly. **Copy its
structure**, swap the model logic:

- `~/git/stem-separator/run_api.py` — entrypoint: read env, load model at startup
  (with `[startup]` log lines), `create_app(...)`, `uvicorn.run(..., workers=1)`.
- `~/git/stem-separator/app/server.py` — the FastAPI app: every endpoint above,
  `require_key`, `_to_status`, `_content_type`, park/unpark wiring. **This is your
  template — your `server.py` will look ~90% identical.**
- `~/git/stem-separator/app/worker.py` — `Job` dataclass + `JobManager`: in-memory
  queue, single daemon worker thread, `_gpu` lock serializing generation vs
  park/unpark, `queue_position`. **Copy this almost verbatim.**
- `~/git/stem-separator/app/schemas.py` — pydantic request/response models. Adapt
  the request to YuE's inputs (§3); keep `Artifact`, `JobStatus`, `JobCreated`,
  `ServiceInfo`, `JobState` as-is.
- `~/git/stem-separator/app/separation.py` — for the **park/unpark pattern only**:
  `park_model` = `model.to("cpu"); torch.cuda.empty_cache()`; `unpark_model` =
  `model.to(device)`. YuE's equivalent lives in the pipeline (§4).

Two more conforming forks if you want another angle: `~/git/stable-audio-3-docker`
(single-artifact generation, JSON request) and
`~/git/AudioSlopServer/references/ACE-Step-1.5-inference-server` (mixed-artifact
music gen — closest in spirit to YuE; look at how it maps audio + codes + lyrics +
metas onto `artifacts[]`).

---

## 3. The generate request (YuE2 inputs)

YuE2 is **text-to-music**: style + lyrics → a symbolic plan (ABC score) → semantic
tokens → audio. **No audio upload** — the request is plain JSON (simpler than
demucs' multipart). Model these fields (source: `src/yue2/cli.py` allowed keys and
`YuE2Pipeline._request`):

```
GenerateRequest {
  style: str            # e.g. "Mandarin, warm piano, acoustic pop, female vocal"
  lyrics: str           # structured lyrics, "[Verse]\n...\n[Chorus]\n..."
  tags: str | None      # alt to style (older prompt form); style preferred
  cot: "full"|"melody"|"off" | None   # chain-of-thought planning depth
  seed: int | None
  cfg_scale: float | None
  abc: str | None       # optional pre-supplied ABC score (skip planning)
  id: str | None        # caller's identity string for the request
  abc_sampling: dict | None       # sampling knobs for the plan phase
  semantic_sampling: dict | None  # sampling knobs for the semantic phase
  file_format: "flac"|"wav" = "flac"   # audio delivery format
  stage: "plan"|"audio" = "audio"      # "plan" = ABC only (fast); "audio" = full song
}
```

Validate with pydantic; reject unknown/invalid with 422. Fill unset knobs from
server-configured defaults (mirror how the reference fills `shifts`/`overlap`).

---

## 4. The YuE2 pipeline (your model calls)

Everything is in `src/yue2/pipeline.py`. Key surface (already inspected):

```python
from yue2.pipeline import YuE2Pipeline

# Load once at startup:
pipe = YuE2Pipeline.from_pretrained(
    model="m-a-p/YuE2-3B",     # DiT/LM weights (HF hub)
    vae="standard",            # "standard" | "legacy" | local path | HF repo (m-a-p/YuE2-Vae)
    device="auto",             # -> cuda when available
    memory_budget_gib=24,      # cap; sets torch per-process memory fraction
    backend="torch",           # USE torch, NOT vllm (see note below)
    quantization="none",       # "fp8" optional
    revision=None,
)

# Full generation in one call (what the CLI's `pipe(**kwargs)` does):
result = pipe(style=..., lyrics=..., tags=..., cot=..., seed=..., cfg_scale=...,
              abc=..., id=..., abc_sampling=..., semantic_sampling=...)   # -> SongResult

# Or stepwise (use if you support stage="plan"):
plan     = pipe.plan(style=..., lyrics=..., tags=...)          # -> SymbolicPlan (ABC)
semantic = pipe.generate_semantic(plan)                        # -> SemanticResult
result   = pipe.synthesize(semantic)                           # -> SongResult
```

`SongResult` (see `pipeline.py`) gives you:
- `.audio` (np.ndarray) + `.sample_rate`
- `.abc` — the ABC score text (`str`)
- `.save(path)` — writes `.flac`/`.wav` audio
- `.save_artifacts(directory)` — writes the whole reproducible bundle:
  `audio.flac`, `score.abc`, `abc_tokens.npy`, `prefix.npy`, `semantic.npy`,
  `latent.npy`, `request.json`, `config.json`, `result.json`, and returns a
  `result` dict (status, identity, audio_seconds, timing, artifact hashes).

**Recommended worker flow:** call `pipe(**request_kwargs)` to get the `SongResult`,
then `result.save_artifacts(job_dir)`, then enumerate the files in `job_dir` into
the contract's `artifacts[]` (§5). For `stage="plan"`, stop after `pipe.plan(...)`
and emit just `score.abc`.

### Park / unpark for YuE
The pipeline already moves the model on and off the GPU (`_load_model` does
`self._model.to(self.device)`; `decode()` does `self._model.to("cpu")` +
`torch.cuda.empty_cache()`; `close()` does the same). Implement:
- `park()`  → `pipe._model.to("cpu"); torch.cuda.empty_cache()` (and the VAE if
  resident). **`empty_cache()` is mandatory** or VRAM never frees.
- `unpark()` → move the model back to `pipe.device`.

Wrap both behind small functions (mirror `park_model`/`unpark_model`) and guard
them with the worker's `_gpu` lock so a park can't race a running generation.
Prefer using a real pipeline method if one exists (`close()` is close to park but
also drops references — don't use it for park; you need the model back). If the
private `_model` handle feels fragile, add a tiny public `park()`/`unpark()` to
`YuE2Pipeline` in this fork — we own it, so that's allowed and cleaner.

### backend = torch, not vllm — deliberate
vllm greedily pins VRAM for its KV cache, which fights ASS's park/share model
(same reason ACE-Step runs its LM on the `pt` backend, not vllm). Default the
image to `backend="torch"`. Leave the `fast` extra (vllm/triton) out of the
default image; a future "speed over shareability" variant can add it.

---

## 5. Artifacts (the mixed bag — YuE's whole point)

Per the ASS contract, a job yields **zero-to-many named, typed artifacts**. YuE is
the canonical mixed-kind case: audio **plus** a score **plus** metadata, not all
audio. Emit (from the files `save_artifacts` wrote):

| name | kind | content_type | notes |
|---|---|---|---|
| `audio.flac` (or `.wav`) | `audio` | `audio/flac` / `audio/wav` | the song |
| `score.abc` | `score` | `text/vnd.abc` (fallback `text/plain`) | the ABC plan |
| `lyrics.txt` | `lyrics` | `text/plain` | the request lyrics, echoed for the record (write it yourself; save_artifacts stores them inside request.json) |
| `result.json` | `metadata` | `application/json` | timing, hashes, audio_seconds |
| `config.json` | `metadata` | `application/json` | resolved config |
| `semantic.npy`, `latent.npy`, `abc_tokens.npy`, `prefix.npy` | `metadata` | `application/octet-stream` | optional; expose for full reproducibility, or skip to keep the list clean |

Give each artifact a real `content_type` (a MIME map like the reference's
`_CONTENT_TYPES`, extended for `.abc`, `.json`, `.npy`, `.txt`). `bytes` = on-disk
size. `kind` is advisory so a client can ask for "the audio one" without hardcoding
filenames. **Minimum viable set: `audio.flac` + `score.abc`** (+ `lyrics.txt`);
expose the rest if cheap.

---

## 6. Dockerfile — OPTIMISE IT (this is a hard requirement)

The single biggest past mistake was Dockerfiles that re-push multi-GB torch/CUDA
layers on every source edit. **Do not repeat it.** Requirements:

1. **Deps BEFORE source.** Copy only the dependency manifests first, install the
   heavy layer, *then* `COPY . .`. A source edit must NOT invalidate the torch
   layer. Pattern:
   ```dockerfile
   COPY pyproject.toml ./
   # (this fork uses setuptools + pyproject; there's no uv.lock. Either pip-install
   #  the pinned deps from pyproject, or generate/commit a lock. Install deps here.)
   RUN pip install --no-cache-dir <the pinned deps>   # big, stable, cached layer
   COPY . .                                            # tiny source layer
   RUN pip install --no-cache-dir --no-deps -e .       # install THIS package only
   ```
   Deps are pinned exactly in `pyproject.toml` (`torch==2.10.0`,
   `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.7.0`,
   `tiktoken==0.12.0`, `numpy==2.2.6`, `soundfile==0.13.1`, `accelerate==1.13.0`)
   plus `fastapi` + `uvicorn[standard]` for the server. Consider a
   `requirements.txt` (or `uv`) so the deps layer is a stable, copyable manifest —
   look at how `~/git/stem-separator/Dockerfile` installs CUDA-matched torch from
   the pytorch index first, then the app requirements. **torch 2.10 needs the
   right CUDA wheel index** — pin it (`--index-url https://download.pytorch.org/whl/cuXXX`)
   like stem-separator does; pick the CUDA that matches torch 2.10.
2. **CUDA base image**, runtime flavour (not devel) unless a build step needs nvcc.
   Match stem-separator's `nvidia/cuda:...-runtime-ubuntu22.04` shape; install
   `python3`, `pip`, `ffmpeg`, `libsndfile1` (soundfile needs it).
3. **Cache env — per-service subdir + one shared token** (ASS convention, current):
   ```dockerfile
   ENV HF_HOME=/cache/huggingface       # generic default; ASS overrides per-service
   ENV TORCH_HOME=/cache/torch          #   to /cache/yue/huggingface etc.
   ENV HF_TOKEN_PATH=/cache/hf-token    # the ONE shared token file, all services
   ```
   Comment that ASS overrides `HF_HOME`/`TORCH_HOME` to `/cache/yue/...` per
   service; the token is shared and never per-service.
4. **`EXPOSE`** the real serving port (§7: `5340`). No stale/aliased ports.
5. **Weights download at STARTUP, announced** — not lazily on first job, and not
   silently. In `run_api.py`, before the server reports ready, load the pipeline
   (which pulls `m-a-p/YuE2-3B` + the VAE from HF on a cold cache — several GB) and
   print clear `[startup] downloading weights (several GB, first run only) — not
   hung, grab a coffee` / `[startup] weights ready` lines. Let HF's progress bars
   through. (This is the exact bug we just fixed in Stable Audio 3 — don't ship it
   again.)
6. Single worker: `CMD` runs `run_api.py`, which calls `uvicorn.run(app,
   host="0.0.0.0", port=<YUE_API_PORT>, workers=1)`. Don't set `--workers >1`.
7. `.dockerignore` the fat stuff (`.git`, `assets`, `examples`, `docs`, `tests`,
   `__pycache__`) so the source layer stays tiny.

**Verify the build succeeds** before releasing. If a manifest references a local
package path, copy the minimum needed for the deps step to resolve.

---

## 7. Wiring into ASS (`ass.toml`)

Add this backend to `~/git/AudioSlopServer/ass.toml` (commit it there too, ASS-side
no rebuild). Follow the existing `[services.acestep]` block as the model — same
per-service cache env + shared token:

```toml
[services.yue]
image = "ghcr.io/stevelittlefish/yue-inference-server:latest"   # match your CI image name
port  = 5340            # YuE. ASS maps host:port -> container:port on the same number;
                        # the container must listen here (YUE_API_PORT below).
verb  = "generate"      # POST /v1/generate; returns audio.flac + score.abc + metadata
evict = "stop"          # START HERE. YuE2-3B is heavy; the parked context tax may not
                        # pay off vs a cold reload. /park + /unpark ARE implemented —
                        # flip to "park" once VRAM/RAM is measured on the box and the
                        # numbers justify it (docs/measurements.md).
ram_reserve_mb = 10000  # rough parked-in-RAM guess for the 3B model; UNMEASURED.
idle_ttl = "0"          # never reclaim parked RAM on the big server
shm_size_mb = 8192      # generation wants elbow room
# This backend reads YUE_API_PORT, not the generic PORT ASS injects, so pin it.
# HF_HOME/TORCH_HOME point at THIS service's own /cache/yue subdir; HF_TOKEN_PATH is
# the one shared token every backend reads.
env = { YUE_API_PORT = "5340", HF_HOME = "/cache/yue/huggingface", TORCH_HOME = "/cache/yue/torch", HF_TOKEN_PATH = "/cache/hf-token" }
volumes = ["/srv/ass/cache:/cache", "/srv/ass/outputs/yue:/app/outputs"]
```

Env-var naming for the server (mirror the reference's `SEP_*`): `YUE_MODEL`
(default `m-a-p/YuE2-3B`), `YUE_VAE` (default `standard`), `YUE_API_PORT`
(default 5340), `YUE_OUTPUT_ROOT` (default `/app/outputs/jobs`), `YUE_API_KEY`
(optional), `YUE_BACKEND` (default `torch`), `YUE_MEMORY_BUDGET_GIB` (default 24),
plus defaults for any request knob you want server-overridable.

**Gated weights?** `m-a-p/YuE2-3B` may require accepting terms on HF. If so, the
shared token at `/cache/hf-token` (via `HF_TOKEN_PATH`) authenticates it — the host
seeds that file once (`printf '%s' hf_xxx > /srv/ass/cache/hf-token`). Confirm
whether the repo is gated and note it.

---

## 8. Release (publish the image)

The other forks cut a `v*` tag that fires CI → GHCR. Set up the same:
- Ensure a container CI workflow exists (copy `.github/workflows/container.yml`
  from `~/git/AudioSlopServer/references/ACE-Step-1.5-inference-server` or
  `~/git/stem-separator`) publishing to
  `ghcr.io/stevelittlefish/yue-inference-server`.
- Copy `make_release.sh` from ACE-Step (`./make_release.sh vX.Y.Z "msg"` → tags +
  pushes → CI builds). Cutting the tag is a human step (a release classifier blocks
  the agent from tag-pushing), so leave clear instructions.

---

## 9. Build order (suggested)

1. `schemas.py` — request + the shared response models (copy reference, adapt request).
2. `app/pipeline_runner.py` (or reuse `yue2.pipeline` directly) — thin wrappers:
   `load_pipeline(env)`, `run_generation(pipe, request, job_dir) -> list[{name,path}]`,
   `park(pipe)`, `unpark(pipe, device)`.
3. `app/worker.py` — copy the reference `Job` + `JobManager` almost verbatim; swap
   `run_separation` for `run_generation`, `park_model`/`unpark_model` for yours.
4. `app/server.py` — copy the reference; change verb to `generate`, request type,
   `_to_status` artifact mapping (kinds per §5), `_CONTENT_TYPES` extended.
5. `run_api.py` — env parsing, announced startup weight load, `create_app`,
   `uvicorn.run(workers=1)`.
6. `Dockerfile` + `.dockerignore` — §6, optimised.
7. Local sanity: `pip install -e .`, run `run_api.py` on CPU if a GPU isn't handy
   (it'll be slow/again may need cuda — at minimum import-check and hit `/health`,
   `/v1/info`, a 422 on a bad request). Then the real test is on the GPU box.
8. `ass.toml` snippet (§7) into `~/git/AudioSlopServer`, committed there.
9. CI workflow + `make_release.sh` (§8). Commit, push, announce
   "Slopping it straight to main!". Leave the tag-push + on-box smoke-test as the
   human's final step.

## 10. Gotchas / decisions to record as you go
- **Single worker, GPU lock.** Never let generation and park/unpark touch CUDA at
  once. The reference's `_gpu` lock handles it — keep it.
- **`/health` gates on weights loaded.** ASS waits on it before sending work; a
  200 before the model is ready causes a first-request crash.
- **Don't vendor upstream torch/vllm logic into ASS** — the boundary is HTTP. You
  only touch this fork and the `ass.toml` snippet.
- **torch 2.10 + CUDA wheel index** must match, or you'll pull CPU wheels and the
  GPU sits idle. Pin the index in the Dockerfile.
- Confirm whether `m-a-p/YuE2-3B` is gated; record the answer near the token note.
- Measure VRAM/RAM on the box before flipping `evict` to `park` — until then,
  `stop` is the safe default.

When it runs green on the box (`/test/yue` test page in ASS, one full song), YuE is
a live ASS backend. Fold any durable notes into ASS's `TODO.md` and delete nothing
until the human confirms.
