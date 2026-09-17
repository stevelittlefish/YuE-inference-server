"""FastAPI application for the YuE2 music-generation service.

~90% identical to the stem-separator reference: the verb is `generate`, the
request is JSON (no upload), and artifacts are a mixed bag (audio + score +
metadata) rather than all-audio stems.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .schemas import Artifact, GenerateRequest, JobCreated, JobStatus, ServiceInfo
from .worker import Job, JobManager

# Map an output file extension to its real MIME type so the contract's
# `content_type` isn't a lie. YuE emits a mixed bag — audio, an ABC score, plain
# text and JSON/npy metadata — and ASS treats content_type as authoritative.
_CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".abc": "text/vnd.abc",
    ".json": "application/json",
    ".txt": "text/plain",
    ".npy": "application/octet-stream",
}

# The advisory `kind` for each artifact, so a client can ask for "the audio one"
# without hardcoding filenames. Anything not named here is metadata.
_KINDS = {
    "audio.flac": "audio",
    "audio.wav": "audio",
    "score.abc": "score",
    "lyrics.txt": "lyrics",
}


def _content_type(path: str) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def _kind(name: str) -> str:
    return _KINDS.get(name, "metadata")


def _to_status(job: Job, manager: JobManager) -> JobStatus:
    artifacts = [
        Artifact(
            name=a["name"],
            kind=_kind(a["name"]),
            content_type=_content_type(a["path"]),
            bytes=Path(a["path"]).stat().st_size if Path(a["path"]).exists() else 0,
        )
        for a in job.artifacts
    ]
    return JobStatus(
        job_id=job.job_id,
        state=job.state,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        queue_position=manager.queue_position(job.job_id),
        request=job.request,
        artifacts=artifacts,
    )


def create_app(
    pipe,
    model_name: str,
    device: str,
    output_root: Path,
    sample_rate: int = 48000,
    backend: str = "torch",
    api_key: Optional[str] = None,
    default_cot: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title="YuE2 Music Generation API", version="1.0")
    manager = JobManager(pipe, output_root)

    def require_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
        if not api_key:
            return
        supplied = x_api_key or (authorization[7:] if authorization and authorization.startswith("Bearer ") else None)
        if supplied != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.get("/health")
    def health():
        # A parked backend is still "up" for readiness; it just has no weights on
        # the GPU. ASS unparks it before sending work.
        return {"status": "ok", "parked": manager.is_parked()}

    @app.get("/v1/info", response_model=ServiceInfo, dependencies=[Depends(require_key)])
    def info():
        return ServiceInfo(
            model=model_name,
            device=device,
            sample_rate=sample_rate,
            parked=manager.is_parked(),
            backend=backend,
        )

    # ASS's park/unpark: our fork's addition so ASS can free the GPU for another
    # model without a full container restart (evict = "park"). No auth — same as
    # /health, these are orchestrator-plane, not user-plane.
    @app.post("/park")
    def park():
        manager.park()
        return {"parked": True}

    @app.post("/unpark")
    def unpark():
        manager.unpark()
        return {"parked": False}

    @app.post("/v1/generate", response_model=JobCreated, dependencies=[Depends(require_key)])
    def generate(request: GenerateRequest):
        # pydantic already rejected unknown/malformed fields with a 422. A couple
        # of cross-field rules the model can't express on its own:
        if request.style is None and request.tags is None:
            raise HTTPException(status_code=422, detail="Provide style (or its alias tags).")
        # Fill unset knobs from the server-wide defaults, so the stored request
        # reflects the values the job actually ran with.
        if request.cot is None and default_cot is not None:
            request.cot = default_cot
        job = manager.submit(request)
        return JobCreated(job_id=job.job_id, state=job.state)

    @app.get("/v1/jobs", response_model=List[JobStatus], dependencies=[Depends(require_key)])
    def list_jobs():
        return [_to_status(j, manager) for j in manager.list()]

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_key)])
    def get_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _to_status(job, manager)

    @app.get("/v1/jobs/{job_id}/result/{name}", dependencies=[Depends(require_key)])
    def get_result(job_id: str, name: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        for a in job.artifacts:
            if a["name"] == name and Path(a["path"]).exists():
                return FileResponse(a["path"], filename=Path(a["path"]).name)
        raise HTTPException(status_code=404, detail=f"Artifact {name!r} not available.")

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def delete_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        shutil.rmtree(job.job_dir, ignore_errors=True)
        manager.remove(job_id)
        return {"deleted": job_id}

    return app
