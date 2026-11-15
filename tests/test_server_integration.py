"""CPU coverage of the fork's HTTP wrapper against the upstream pipeline.

Reuse upstream's tiny local checkpoints: no GPU or published model download.
"""
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from app import pipeline_runner, worker
from app.schemas import GenerateRequest
from app.server import create_app
from yue2 import pipeline
from yue2.pipeline import SongResult, SymbolicPlan
from test_offline_workflow import offline_models, open_pipeline


def client_for(pipe, output_root, **kwargs):
    return TestClient(create_app(
        pipe, model_name="local-test-model", device="cpu",
        output_root=output_root, **kwargs,
    ))


def completed_job(client, request):
    response = client.post("/v1/generate", json=request)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = client.get(f"/v1/jobs/{job_id}")
        assert status.status_code == 200
        job = status.json()
        if job["state"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.01)
    pytest.fail(f"Job {job_id} did not complete")


@pytest.mark.parametrize("file_format", ["flac", "wav"])
def test_http_audio_roundtrip_and_native_result_integrity(offline_models, tmp_path, file_format):
    with open_pipeline(offline_models) as pipe, client_for(pipe, tmp_path) as client:
        job = completed_job(client, {
            "style": "piano", "lyrics": "La la", "cot": "full", "seed": 42,
            "abc": "X:1\r\nM:4/4\r\nK:C\r\nCDEF|\r\n", "file_format": file_format,
        })
        assert job["state"] == "succeeded", job["error"]
        assert job["artifacts"][0]["name"] == f"audio.{file_format}"
        assert client.get("/v1/jobs").json()[0]["job_id"] == job["job_id"]
        directory = tmp_path / job["job_id"]
        restored = SongResult.load(directory)
        assert restored.sample_rate == 48000
        assert restored.audio.shape[1] == 2
        assert restored.semantic.plan.request.seed == 42
        assert (directory / "lyrics.txt").read_text() == "La la"
        for artifact in job["artifacts"]:
            downloaded = client.get(f"/v1/jobs/{job['job_id']}/result/{artifact['name']}")
            assert downloaded.status_code == 200
            assert downloaded.content == (directory / artifact["name"]).read_bytes()
            assert len(downloaded.content) == artifact["bytes"]
        assert client.get(f"/v1/jobs/{job['job_id']}/result/missing").status_code == 404
        assert all(p.device.type == "cpu" for p in pipe._model.parameters())
        assert all(p.device.type == "cpu" for p in pipe._vae.parameters())


def test_http_plan_generates_only_score_and_records_random_seed(offline_models, tmp_path):
    with open_pipeline(offline_models) as pipe, client_for(pipe, tmp_path) as client:
        job = completed_job(client, {
            "style": "piano", "lyrics": "La la", "cot": "full", "stage": "plan",
            "abc_sampling": {"temperature": 0, "min_tokens": 1, "max_tokens": 2},
        })
        assert job["state"] == "succeeded", job["error"]
        directory = tmp_path / job["job_id"]
        plan = SymbolicPlan.load(directory)
        assert len(plan.abc_ids) == 2  # Tiny zero-logit model hits the token budget.
        assert plan.truncated
        assert plan.request.seed == job["request"]["seed"]
        assert 0 <= plan.request.seed < 2**63
        assert not (directory / "semantic.npy").exists()
        assert not (directory / "latent.npy").exists()
        assert not list(directory.glob("audio.*"))
        assert pipe._vae is None
        assert client.delete(f"/v1/jobs/{job['job_id']}").status_code == 200
        assert not directory.exists()
        assert client.get(f"/v1/jobs/{job['job_id']}").status_code == 404


def test_http_failed_job_does_not_prevent_next_job(offline_models, tmp_path):
    with open_pipeline(offline_models) as pipe, client_for(pipe, tmp_path) as client:
        failed = completed_job(client, {
            "style": "piano", "lyrics": "La", "cot": "off", "abc": "invalid combination",
        })
        assert failed["state"] == "failed"
        assert "External ABC" in failed["error"]
        good = completed_job(client, {
            "style": "piano", "lyrics": "La", "stage": "plan", "cot": "full",
            "abc": "X:1\nK:C\nCDEF|", "seed": 7,
        })
        assert good["state"] == "succeeded", good["error"]


def test_http_auth_validation_health_and_lazy_unpark(tmp_path):
    moves = []
    pipe = SimpleNamespace(park=lambda: moves.append("park"),
                           unpark=lambda: pytest.fail("server unpark must stay lazy"))
    with client_for(pipe, tmp_path, api_key="test-key") as client:
        assert client.get("/health").json() == {"status": "ok", "parked": False}
        assert client.get("/v1/info").status_code == 401
        assert client.post("/park").json() == {"parked": True}
        assert client.post("/park").status_code == 200
        assert moves == ["park"]
        assert client.get("/health").json()["parked"]
        assert client.post("/unpark").json() == {"parked": False}
        assert client.post("/unpark").status_code == 200
        client.headers["X-API-Key"] = "test-key"
        info = client.get("/v1/info").json()
        assert info["parked"] is False
        assert info["stages"] == ["plan", "audio"]
        assert "cuda" in info["vram"]
        for request in ({"lyrics": "La"}, {"style": "piano", "lyrics": "La", "unknown": 1}):
            assert client.post("/v1/generate", json=request).status_code == 422


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "xpu:0", "mps"])
def test_park_keeps_weights_and_clears_selected_device(monkeypatch, device):
    moves, cleared = [], []
    pipe = pipeline.YuE2Pipeline.__new__(pipeline.YuE2Pipeline)
    pipe.device = torch.device(device)
    model = SimpleNamespace(to=lambda target: moves.append(("model", str(target))))
    vae = SimpleNamespace(to=lambda target: moves.append(("vae", str(target))))
    pipe._model, pipe._vae = model, vae
    monkeypatch.setattr(pipeline, "empty_cache", lambda target: cleared.append(str(target)))
    pipe.park()
    assert moves == [("model", "cpu"), ("vae", "cpu")]
    assert cleared == [device]
    assert pipe._model is model and pipe._vae is vae
    pipeline_runner.unpark(pipe)
    assert len(moves) == 2  # Wrapper deliberately leaves reload to the next phase.
    pipe._model = pipe._vae = None
    pipe.park()  # Safe even when models have not loaded.


def test_preload_uses_local_model_and_cpu_decoder(offline_models):
    with open_pipeline(offline_models) as pipe:
        pipeline_runner.preload(pipe)
        model, vae = pipe._model, pipe._vae
        assert model is not None and vae is not None
        assert next(model.parameters()).device.type == "cpu"
        assert next(vae.parameters()).device.type == "cpu"
        assert vae.decoder_only and not hasattr(vae, "encoder")
        pipeline_runner.preload(pipe)
        assert pipe._model is model and pipe._vae is vae


def test_park_waits_for_active_generation(monkeypatch, tmp_path):
    generating, release, parking, parked = (threading.Event() for _ in range(4))

    def generate(*args):
        generating.set()
        assert release.wait(5), "generation was not released"
        return []

    monkeypatch.setattr(worker, "run_generation", generate)
    pipe = SimpleNamespace(park=parked.set)
    manager = worker.JobManager(pipe, tmp_path)
    job = manager.submit(GenerateRequest(style="piano", lyrics="La"))
    assert generating.wait(5)

    def park():
        parking.set()
        manager.park()

    thread = threading.Thread(target=park)
    thread.start()
    try:
        assert parking.wait(5)
        assert not parked.wait(0.05), "park ran while generation held the GPU lock"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert parked.is_set()
    assert job.error is None


def test_startup_preloads_before_serving_and_reports_pipeline_device(monkeypatch, tmp_path):
    import run_api

    events = []
    pipe = SimpleNamespace(device=torch.device("xpu:0"))
    app = object()

    def load():
        events.append("load")
        return pipe

    def preload(actual):
        assert actual is pipe
        events.append("preload")

    def create(**kwargs):
        assert kwargs["pipe"] is pipe
        assert kwargs["device"] == "xpu:0"
        events.append("app")
        return app

    def serve(actual, **kwargs):
        assert actual is app
        assert kwargs["workers"] == 1
        events.append("serve")

    monkeypatch.setenv("YUE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(run_api, "load_pipeline", load)
    monkeypatch.setattr(run_api, "preload", preload)
    monkeypatch.setattr(run_api, "create_app", create)
    monkeypatch.setattr(run_api.uvicorn, "run", serve)
    run_api.main()
    assert events == ["load", "preload", "app", "serve"]
