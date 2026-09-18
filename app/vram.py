"""GPU memory self-report for ASS.

ASS owns the VRAM budget but deliberately holds no CUDA context of its own, so it
can't read the card directly. Each backend can, though — torch is already loaded
in-process — so we expose our own usage in /v1/info and let ASS record it. The key
number is peak_mb: torch.cuda.max_memory_allocated tracks the high-water mark since
the process started, so a single read (even after a job finishes) still captures
the peak that happened DURING inference. No live polling, no nvidia-smi, no new
dependency — just three torch counters that are already there.
"""

from __future__ import annotations


def vram_stats() -> dict:
    """Current + peak GPU memory for this process, in MB. Returns {"cuda": False}
    when there's no CUDA (the GPU-less dev box), so callers can always include it."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
    except Exception:
        return {"cuda": False}

    mb = 1 << 20
    idx = torch.cuda.current_device()
    return {
        "cuda": True,
        "device": f"cuda:{idx}",
        "allocated_mb": torch.cuda.memory_allocated() // mb,
        "reserved_mb": torch.cuda.memory_reserved() // mb,
        "peak_mb": torch.cuda.max_memory_allocated() // mb,
    }
