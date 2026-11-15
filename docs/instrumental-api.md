# Instrumentals over HTTP

The API accepts `instrumental: true` on an ordinary generation request. The
worker handles composition, score conversion and audio rendering as **one job**.
A Go client needs only its usual JSON submission, polling and artifact downloads.
ASS forwards the request unchanged; no ASS orchestrator or configuration change
is required. The existing ASS song form has not gained an instrumental control;
send JSON from your client to use this option.

```bash
curl http://localhost:5340/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "instrumental": true,
    "style": "Warm solo piano, gentle and lyrical, 90 BPM",
    "seed": 42
  }'
```

Through ASS, send that same JSON to `POST /v1/yue/jobs` on the ASS address
(normally port 2645). Poll `GET /v1/jobs/{job_id}` and download artifacts at
`GET /v1/jobs/{job_id}/result/{name}`, as for an ordinary song.

## Request behavior

| Field | Instrumental behavior |
| --- | --- |
| `instrumental` | Set to `true`; default `false` preserves song generation. |
| `style` / `tags` | Supply one, or matching values. Instrumental/no-vocal conditions are added automatically. |
| `lyrics` | May be omitted, empty, or section tags only. Actual sung words are rejected with HTTP 422. |
| `cot` | `full` or `melody`; defaults to `full` for instrumentals, even if the server song default is `off`. Explicit `off` is rejected with HTTP 422. |
| `abc` | Optional native two-voice ABC. If supplied, skip composition and convert this score. |
| `stage` | `audio` by default; `plan` returns the converted score and evidence without running audio generation. |
| `seed` | Optional; a blank seed is randomized once and shared across composition and rendering. |
| Sampling / guidance | Existing `abc_sampling`, `semantic_sampling` and `cfg_scale` fields remain available. |
| `file_format` | Existing `flac` / `wav` behavior, including native FLAC retention. |

When lyrics are omitted or blank, the planner receives Intro, Verse, Chorus and
Outro section tags, without sung words. To suggest a different form, provide
tags such as `"[Intro]\n\n[Verse]\n\n[Outro]\n"`. Supported tags are Intro, Verse,
Pre-Chorus, Chorus, Bridge, Interlude, Outro and Instrumental (case-insensitive).
Rendering always receives empty lyrics, including when planner tags are supplied.

Ordinary song requests still require nonempty lyrics. Detect support through
`GET /v1/info`: updated backends return `"supports_instrumental": true`. Older
backends reject the new request field.

## Inside the worker

1. Generate a native score with the resident YuE2 pipeline, or take the supplied
   ABC. Reject an empty or truncated plan before starting acoustic generation.
2. Use upstream's `instrumentalize.py` to transfer Vocal notes to Ins, preserving
   their pitches, onsets and durations. Vocal-priority overlap handling can
   reduce existing Ins notes; the transfer report records each affected note.
3. Use upstream's `instrumental.py prepare` to validate the converted score,
   add the instrumental style conditions and choose `full` when the score has
   chords or `melody` otherwise. The render mode follows the actual score,
   regardless of the planning mode selected in the request.
4. Render that exact converted ABC through the resident pipeline, or stop after
   saving it for `stage="plan"`. The normal GPU lock covers the whole job.

The score tools run as short-lived, CPU-only Python subprocesses with a 30-second
timeout per tool. They do not load models. This isolates their generic Python
module names from the server and reuses upstream's conversion implementation
without changing `src/yue2` or the bundled skill. Model work stays in the existing
process. Invalid or unsupported native notation fails the job; it does not fall
back to a newly composed score or regular vocal rendering.

## Artifacts and reproducibility

The normal audio, score, tokens, latents and metadata are retained. Instrumental
jobs additionally publish:

| Artifact | Contents |
| --- | --- |
| `original_score.abc` | The generated or supplied score, before voice transfer. |
| `original_plan.json` | Planner request, exact score/prefix IDs, timings and truncation flag. |
| `instrumental.json` | Original API request, selected planning/render modes, transfer report and score validation results. |
| `score.abc` | The converted score actually supplied to rendering. |

`request.json` in an audio job records the **effective render request**: converted
ABC, empty lyrics, augmented style and selected render mode. The status endpoint
and `instrumental.json` preserve the original API input, with the resolved seed.
The original score and instrumental evidence are included in the native audio
bundle's artifact hashes, so `SongResult.load()` can verify the complete result.
`original_plan.json` is diagnostic metadata, not a directory loadable by
`SymbolicPlan.load()`; the final saved plan retains the usual loading interface.

Job timestamps cover the complete workflow. Composition timings are in
`original_plan.json`; the native `result.json` timings cover rendering from the
converted score. A failed job retains partial files on the server for diagnosis,
but only completed jobs publish downloadable artifacts under the current contract.

## Validation scope

CPU integration tests exercise the real HTTP worker, upstream score conversion,
validation, music sampling, flow matching, decoding and native bundle loading
with tiny local checkpoints. Tests of newly composed scores supply deterministic
ABC at the planner's token-output boundary because random tiny checkpoints
cannot compose meaningful notation. Tests also cover supplied scores, plan-only
jobs, overlap reporting, section tags, malformed/truncated scores and recovery.

The complete CPU regression suite passed with **436 tests, 25 hardware-dependent
skips and 30 passing subtests** on Python 3.12.16 / PyTorch 2.10.0+cpu. The new
instrumental tests are in `tests/test_instrumental_api.py`; use the offline test
environment described in [upstream integration](upstream-integration.md).

This checks plumbing and symbolic invariants. Full-checkpoint GPU generation and
listening are still required to assess the result. Instrumental conditioning does
not guarantee the absence of vocals or exact waveform timing.
