"""Instrumental HTTP jobs using upstream score tools and tiny CPU models."""
import json
import subprocess

import pytest

from app import instrumental
from yue2.pipeline import SongResult, SymbolicPlan
from test_offline_workflow import offline_models, open_pipeline
from test_server_integration import client_for, completed_job


NATIVE_SCORE = '''X:1
T:
M:4/4
L:1/32
Q:1/4=96
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
"C"C8D8E8G8|
V: Ins
c32|
'''


@pytest.mark.parametrize("supplied,stage,chords,file_format", [
    (False, "audio", True, "flac"),
    (True, "audio", True, "wav"),
    (False, "plan", True, "flac"),
    (True, "plan", False, "flac"),
])
def test_one_http_job_converts_and_renders_instrumental(
    offline_models, tmp_path, monkeypatch, supplied, stage, chords, file_format,
):
    score = NATIVE_SCORE if chords else NATIVE_SCORE.replace('"C"', '')
    # Tiny random checkpoints cannot compose valid notation. Supply deterministic
    # ABC output only; run actual plan construction, tools, music sampling, flow
    # matching, decoder, worker and artifact handling with the local checkpoints.
    with open_pipeline(offline_models) as pipe:
        generate = pipe._generate
        phases = []

        def generate_tokens(prefix, sampling, seed, phase, **kwargs):
            phases.append(phase)
            if phase == "abc":
                return pipe.tokenizer.encode(score), {"seconds": 0.1}, False
            return generate(prefix, sampling, seed, phase, **kwargs)

        monkeypatch.setattr(pipe, "_generate", generate_tokens)
        with client_for(pipe, tmp_path, default_cot="off") as client:
            assert client.get("/v1/info").json()["supports_instrumental"] is True
            request = {"instrumental": True, "tags": "warm piano", "stage": stage,
                       "file_format": file_format, "id": "i" * 150}
            if supplied:
                request["abc"] = score
                request["lyrics"] = ""  # Omitted and explicit empty are supported.
            job = completed_job(client, request)
            assert job["state"] == "succeeded", job["error"]
            assert job["request"]["instrumental"] is True
            assert job["request"]["cot"] == "full"  # Ignores server default off.
            directory = tmp_path / job["job_id"]
            assert (directory / "original_score.abc").read_text() == score
            original_plan = json.loads((directory / "original_plan.json").read_text())
            assert original_plan["truncated"] is False
            assert original_plan["request"]["lyrics"].startswith("[Intro]")
            assert original_plan["request"]["style"].startswith("Instrumental,")
            report = json.loads((directory / "instrumental.json").read_text())
            assert report["source"] == ("provided" if supplied else "generated")
            assert report["api_request"]["instrumental"] is True
            assert report["api_request"]["seed"] == job["request"]["seed"]
            assert report["transfer"]["vocal_pitch_onset_duration_preserved"] is True
            assert report["transfer"]["vocal_notes_before"] == 4
            assert report["transfer"]["vocal_notes_after"] == 0
            assert len(report["transfer"]["affected_ins_notes"]) == 1
            voices = report["score_check"]["voices"]
            assert voices["Vocal"]["sounding_notes"] == 0
            assert [note["midi_pitch"] for note in voices["Ins"]["notes"]] == [60, 62, 64, 67]
            plan = SymbolicPlan.load(directory)
            assert plan.request.lyrics == ""
            assert plan.request.id == "i" * 150
            assert plan.request.cot == ("full" if chords else "melody")
            assert (directory / "lyrics.txt").read_text() == ""
            assert phases.count("abc") == (0 if supplied else 1)
            if stage == "audio":
                result = SongResult.load(directory)
                assert result.audio.shape[1] == 2
                assert result.latents.shape == (3, 64)
                assert phases.count("semantic") == 1
                assert job["artifacts"][0]["name"] == f"audio.{file_format}"
                # Evidence is covered by the native integrity receipt.
                receipt = json.loads((directory / "result.json").read_text())
                assert {"original_score.abc", "original_plan.json", "instrumental.json"} <= receipt["artifacts"].keys()
            else:
                assert phases.count("semantic") == 0
                assert pipe._vae is None
                assert not list(directory.glob("audio.*"))
            original_artifact = next(a for a in job["artifacts"] if a["name"] == "original_score.abc")
            assert original_artifact["kind"] == "score"
            for name in ("original_score.abc", "score.abc", "instrumental.json"):
                response = client.get(f"/v1/jobs/{job['job_id']}/result/{name}")
                assert response.status_code == 200
                assert response.content == (directory / name).read_bytes()
            assert not list(directory.glob("instrumental-*"))


@pytest.mark.parametrize("extra", [
    {}, {"lyrics": ""},
    {"instrumental": True, "lyrics": "Words to sing"},
    {"instrumental": True, "lyrics": "[Verse]\nWords to sing"},
    {"instrumental": True, "lyrics": "[Unrecognised]"},
    {"instrumental": True, "cot": "off"},
    {"instrumental": True, "cot": "invalid"},
    {"instrumental": True, "tags": "conflicting style"},
])
def test_bad_instrumental_combinations_are_rejected_before_queueing(tmp_path, extra):
    with client_for(object(), tmp_path) as client:
        response = client.post("/v1/generate", json={"style": "piano", **extra})
        assert response.status_code == 422
        assert client.get("/v1/jobs").json() == []


@pytest.mark.parametrize("output", ["empty", "truncated", "invalid"])
def test_bad_model_score_fails_before_music_generation_and_worker_recovers(
    offline_models, tmp_path, monkeypatch, output,
):
    with open_pipeline(offline_models) as pipe:
        phases = []

        def generate_tokens(prefix, sampling, seed, phase, **kwargs):
            phases.append(phase)
            assert phase == "abc", "invalid plan reached audio generation"
            text = "" if output == "empty" else "bad ABC" if output == "invalid" else NATIVE_SCORE
            return pipe.tokenizer.encode(text), {}, output == "truncated"

        monkeypatch.setattr(pipe, "_generate", generate_tokens)
        with client_for(pipe, tmp_path) as client:
            failed = completed_job(client, {"style": "piano", "instrumental": True})
            assert failed["state"] == "failed"
            assert "Instrumental" in failed["error"]
            assert phases == ["abc"]
            failed_dir = tmp_path / failed["job_id"]
            assert (failed_dir / "original_plan.json").exists()
            assert not list(failed_dir.glob("instrumental-*"))
            assert not (failed_dir / "semantic.npy").exists()
            good = completed_job(client, {"style": "piano", "instrumental": True,
                                          "abc": NATIVE_SCORE, "stage": "plan"})
            assert good["state"] == "succeeded", good["error"]


def test_section_tags_are_planner_only_and_source_line_endings_survive(
    offline_models, tmp_path,
):
    score = NATIVE_SCORE.replace("\n", "\r\n")
    with open_pipeline(offline_models) as pipe, client_for(pipe, tmp_path) as client:
        job = completed_job(client, {"style": "piano", "instrumental": True,
                                     "lyrics": "[Verse]\n", "abc": score, "stage": "plan"})
        assert job["state"] == "succeeded", job["error"]
        directory = tmp_path / job["job_id"]
        assert (directory / "original_score.abc").read_bytes() == score.encode()
        original = json.loads((directory / "original_plan.json").read_text())
        assert original["request"]["lyrics"] == "[Verse]\n"
        assert SymbolicPlan.load(directory).request.lyrics == ""


def test_score_tool_timeout_has_a_clear_failure(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(instrumental.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="exceeded 30 seconds"):
        instrumental._run_tool("instrumentalize.py", "unused.abc")
