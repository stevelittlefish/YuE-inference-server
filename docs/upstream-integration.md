# Upstream integration: November 1, 2026

This integration merges upstream `1116bea76032bee5668e5f097294d082fa0b84f0`
into the HTTP-server fork. The previous shared revision was
`bd90e4ccae671d869b3ecaca6d7e893927d29442` (September 17, 2026).

## Fork compatibility

- Retain the server README, HTTP endpoints, serialized worker, randomized blank
  seeds, startup preloading and lazy server unpark behavior.
- Preserve the fork's pipeline `park()` / `unpark()` methods. Parking now uses
  upstream's device-aware cache cleanup, including XPU and MPS as well as CUDA.
- Report the pipeline's resolved device at server startup and in `/v1/info`,
  including upstream's additional accelerator types.
- For WAV requests, retain the native `audio.flac` and list `audio.wav` first.
  Upstream's new `SongResult.load()` and artifact verification require the FLAC
  named in the native result receipt. Deleting it made a successful WAV job's
  saved bundle fail verification.
- Keep the existing HTTP request schema. Upstream's CLI/Python instrumental and
  low-memory workflows are available, but this merge does not add HTTP controls
  for them or relax the API's nonempty-lyrics requirement.

## What CPU validation establishes

Validation uses Python 3.12, PyTorch 2.10.0+cpu and the pinned pipeline
dependencies. Hugging Face is offline and accelerator visibility is disabled.
Tests create small local checkpoints rather than downloading published weights.

Results on November 15, 2026, with Python 3.12.16 and PyTorch 2.10.0+cpu:

| Check | Result |
| --- | --- |
| Combined upstream and HTTP pytest suite | 419 passed, 25 hardware skips, 30 subtests passed |
| Standalone instrumental workflow suite | 51 passed |
| Strict skill-bundle verification and archive build | Passed |
| Offline Python source distribution and wheel build | Passed |
| CLI help and environment doctor | Dependencies ready; no CUDA, MPS or XPU available |

Only dependency deprecation warnings were reported by the passing test suite.

The upstream suite covers model equations, sampling, attention visibility,
AR/NAR placement, cancellation and retry, VAE decoding, artifact integrity,
offline export, CLI behavior and portable skill delivery. Some tests run the
complete music-code → flow matching → VAE → saved-result path with tiny models.

`tests/test_server_integration.py` additionally exercises:

- Actual HTTP submission, worker execution, status polling and byte-for-byte
  artifact downloads for both FLAC and WAV, followed by native result loading.
- Generated-score-only jobs, truncation reporting and recorded random seeds.
- Worker recovery after a failed request, plus job deletion.
- Authentication, malformed-request rejection, readiness and service metadata.
- Startup preloading and reporting of the selected pipeline device.
- Weight preservation and device-cache cleanup when parking, lazy unpark, and
  exclusion between generation and parking. Accelerator transfers are mocked
  for these hardware-independent parking assertions.

To reproduce in a fresh Python 3.12 virtual environment:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[test]'

HF_HOME=/tmp/yue2-test-hf \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest -q

python -B skills/yue2-music/instrumental/scripts/test_tools.py
python -B tools/build_release.py --skill-only --output /tmp/yue2-skill-dist
```

Use a writable `HF_HOME` because Transformers caches reviewed Python code even
when loading local checkpoints offline. Disable bytecode generation before
running tests: strict portable-bundle verification rejects extra files inside
the instrumental bundle, including `__pycache__` entries.

The existing CPU GitHub Actions workflow remains disabled as it was in the
fork; these commands run explicitly. In a restricted sandbox, Starlette's
cross-thread TestClient communication may require execution outside the sandbox.

## What still requires a GPU

Tiny randomly initialized models establish code compatibility, not musical
quality. CPU tests cannot establish full-checkpoint CUDA graph correctness,
BF16 behavior on the target accelerator, peak VRAM, driver compatibility,
model transfer performance, or production generation time. The container's
CUDA installation also needs validation on the deployment machine.

Before tagging a release, run this acceptance check on the intended GPU with
the same model/VAE revisions and memory budget used in production:

1. Build the container with `docker build -t yue2-upstream-check .`, then run
   it using the GPU/cache mounts in the [README](../README.md#docker-recommended).
   Use an isolated port and output location if a production instance is active.
   Wait for `GET /health` to return 200; inspect `/v1/info` and startup logs.
2. Submit a plan-only request with `stage="plan"`, `cot="full"`, style, lyrics
   and a fixed seed. Poll its job status, download `score.abc`, and check the
   plan's truncation flag. Confirm that no audio artifacts were created.
3. Submit [examples/song.json](../examples/song.json) as a normal audio job.
   Wait for success, download and listen to `audio.flac`, inspect `result.json`
   for truncation, finite timings and 48 kHz output. Verify the bundle using
   `yue2 inspect PATH_TO_JOB_DIRECTORY` in the same runtime.
4. Run a second audio job on the same process, requesting `file_format="wav"`.
   Confirm WAV is listed first, both audio files can be downloaded, the native
   bundle verifies, and the second run neither crashes nor exhausts memory.
5. Call `/park` twice. Confirm readiness remains available, `parked` is true,
   and GPU memory is released using `/v1/info` and `nvidia-smi`. Call `/unpark`
   twice, then submit another generation. Unpark is deliberately lazy: weights
   return to the accelerator when the next generation needs them.
6. Compare a fixed request against the previous release on the same machine.
   Record the score, semantic tokens, latents, listening result, stage timings
   and peak VRAM. Upstream attention/kernel changes can alter sampled output,
   so assess complete results rather than requiring identical audio bytes.

These full-model GPU checks have not been performed as part of the CPU merge.
