"""Request/response models for the YuE2 music-generation API.

The response models (Artifact, JobStatus, JobCreated, ServiceInfo, JobState) are
the shared ASS contract shapes — kept identical to the stem-separator reference so
ASS talks to every backend the same way. Only the request is YuE-shaped.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


FILE_FORMATS = ("flac", "wav")
COT_MODES = ("full", "melody", "off")
STAGES = ("plan", "audio")


class GenerateRequest(BaseModel):
    """YuE2 is text-to-music: style + lyrics -> ABC plan -> semantic -> audio.

    No audio upload — this is plain JSON. Unset sampling/cfg knobs fall back to
    the pipeline's own defaults; unset cot/file_format/stage fall back to the
    server-configured defaults (mirror the reference's shifts/overlap fill-in).
    """

    model_config = {"extra": "forbid"}  # reject unknown fields with a 422

    style: Optional[str] = None  # "Mandarin, warm piano, acoustic pop, female vocal"
    lyrics: str = Field(..., min_length=1)  # "[Verse]\n...\n[Chorus]\n..."
    tags: Optional[str] = None  # older alias for style; style preferred, must agree
    cot: Optional[str] = None  # chain-of-thought planning depth: full|melody|off
    seed: Optional[int] = Field(default=None, ge=0, lt=2**63)
    cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    abc: Optional[str] = None  # pre-supplied ABC score (skips planning)
    id: Optional[str] = None  # caller's identity string for the request
    abc_sampling: Optional[Dict[str, Any]] = None  # plan-phase sampling knobs
    semantic_sampling: Optional[Dict[str, Any]] = None  # semantic-phase knobs
    file_format: str = "flac"  # audio delivery format: flac|wav
    stage: str = "audio"  # "plan" = ABC only (fast); "audio" = full song


class Artifact(BaseModel):
    """One named, typed output of a job — the ASS contract's unit of result.

    YuE is the canonical mixed-kind case: audio plus a score plus metadata.
    `content_type` is the real MIME, `kind` an advisory role so a client can ask
    for "the audio one" without hardcoding filenames, `bytes` the on-disk size.
    """

    name: str  # e.g. "audio.flac", "score.abc", "result.json"
    kind: str  # advisory: audio | score | lyrics | metadata
    content_type: str
    bytes: int


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    error: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    queue_position: Optional[int] = None
    request: Optional[GenerateRequest] = None
    artifacts: List[Artifact] = Field(default_factory=list)


class JobCreated(BaseModel):
    job_id: str
    state: JobState


class VRAMStats(BaseModel):
    """This process's GPU memory, in MB. peak_mb is the high-water mark since the
    process started (torch.cuda.max_memory_allocated), so it captures the inference
    peak even when ASS reads it after a job. cuda=False on a GPU-less box."""

    cuda: bool = False
    device: Optional[str] = None
    allocated_mb: int = 0
    reserved_mb: int = 0
    peak_mb: int = 0


class ServiceInfo(BaseModel):
    model: str
    device: str
    sample_rate: int
    # Whether ASS has parked us: weights on CPU, GPU freed. Still "up" for
    # readiness — ASS unparks before sending work.
    parked: bool = False
    backend: str = "torch"
    verb: str = "generate"
    cot_modes: List[str] = Field(default_factory=lambda: list(COT_MODES))
    stages: List[str] = Field(default_factory=lambda: list(STAGES))
    file_formats: List[str] = Field(default_factory=lambda: list(FILE_FORMATS))
    vram: VRAMStats = Field(default_factory=VRAMStats)
