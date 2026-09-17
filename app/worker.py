"""In-memory job queue with a single worker thread.

Generation is serialized on one worker: the model runs one job at a time and the
store is in-memory, so the service must run under a single process
(uvicorn --workers 1). Copied almost verbatim from the stem-separator reference;
run_separation is swapped for run_generation and there is no input file to clean
up (YuE requests are pure JSON).
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .pipeline_runner import park, run_generation, unpark
from .schemas import GenerateRequest, JobState


@dataclasses.dataclass
class Job:
    job_id: str
    request: GenerateRequest
    job_dir: Path
    state: JobState = JobState.queued
    error: Optional[str] = None
    created_at: float = dataclasses.field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Completed outputs as {name, path} dicts; the server layer turns these into
    # the contract's typed Artifact objects (adding kind/content_type/bytes).
    artifacts: List[dict] = dataclasses.field(default_factory=list)


class JobManager:
    def __init__(self, pipe, output_root: Path):
        self._pipe = pipe
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        # The GPU is a single resource: a running generation and a park/unpark
        # must never touch it at once. This lock serializes them. (ASS won't send
        # work during a swap anyway — its lease guarantees it — but the backend
        # shouldn't rely on a caller being well-behaved.)
        self._gpu = threading.Lock()
        self._parked = False
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="yue-worker", daemon=True)
        self._thread.start()

    def park(self) -> None:
        """Move weights to CPU RAM and free the GPU. Idempotent."""
        with self._gpu:
            if self._parked:
                return
            park(self._pipe)
            self._parked = True

    def unpark(self) -> None:
        """Move weights back onto the GPU. Idempotent."""
        with self._gpu:
            if not self._parked:
                return
            unpark(self._pipe)
            self._parked = False

    def is_parked(self) -> bool:
        return self._parked

    def submit(self, request: GenerateRequest) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self._output_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, request=request, job_dir=job_dir)
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in self._order]

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)

    def queue_position(self, job_id: str) -> Optional[int]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state not in (JobState.queued, JobState.running):
                return None
            ahead = 0
            for jid in self._order:
                if jid == job_id:
                    break
                if self._jobs[jid].state in (JobState.queued, JobState.running):
                    ahead += 1
            return ahead

    def _run(self):
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is not None:
                self._process(job)

    def _process(self, job: Job):
        with self._lock:
            job.state = JobState.running
            job.started_at = time.time()
        try:
            # Hold the GPU for the duration so a concurrent /park can't yank the
            # weights to CPU mid-generation.
            with self._gpu:
                artifacts = run_generation(self._pipe, job.request, job.job_dir)
            with self._lock:
                job.artifacts = artifacts
                job.state = JobState.succeeded
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            with self._lock:
                job.state = JobState.failed
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.finished_at = time.time()
