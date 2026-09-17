# YuE2 ASS backend — measurements & deployment notes

Numbers ASS wants before it can make good scheduling decisions. Fill these in on
the actual GPU box; until then the conservative defaults stand.

## Weights are NOT gated (checked 2026-09-18)

`m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`, and `m-a-p/YuE2-Vae-legacy` all report
`gated: false` from the HF API — no terms to accept, no token strictly required
to pull them. The shared token at `HF_TOKEN_PATH=/cache/hf-token` is still read if
present (harmless, and future-proof if the repo is ever gated), but a cold cache
downloads fine without one. If m-a-p flips gating on later, seed the token once on
the host: `printf '%s' hf_xxx > /srv/ass/cache/hf-token`.

## evict: stop vs park — UNMEASURED

`ass.toml` starts YuE at `evict = "stop"`. `/park` + `/unpark` are implemented
(move the model to CPU RAM + `torch.cuda.empty_cache()`, and back). Before
flipping to `evict = "park"`, measure on the box and fill in:

| metric | value | how |
|---|---|---|
| VRAM resident (model loaded) | ? GiB | `nvidia-smi` mid-idle after startup |
| VRAM after `/park` | ? GiB | POST /park, then `nvidia-smi` (should drop to ~0) |
| RAM held while parked | ? GiB | `docker stats` after /park — sets `ram_reserve_mb` |
| cold reload (stop→start→ready) | ? s | container start to first 200 on /health |
| unpark latency | ? s | POST /unpark round-trip |

Park pays off only if (RAM tax + unpark latency) beats a cold reload often enough
given how ASS rotates backends on the box. `ram_reserve_mb = 10000` in `ass.toml`
is a guess for the 3B model parked in RAM — refine it from the table above.
