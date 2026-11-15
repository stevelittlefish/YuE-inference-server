"""API adapter around upstream's portable instrumental tools.

Run their CLIs in subprocesses so their generic module names (common,
abc_tools, compile_score) cannot collide with imports in the serving process.
Only these score operations run here; all model work uses the resident pipeline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .schemas import GenerateRequest


_SCRIPTS = Path(__file__).resolve().parents[1] / "skills/yue2-music/instrumental/scripts"
_DEFAULT_SECTIONS = "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n"
_NO_VOCALS = ("no vocals", "no singing", "no choir", "no spoken words")
# Style phrases that ask for a voice; they would contradict the no-vocal terms.
_VOCAL_PHRASE = re.compile(
    r"\b(vocals?|voices?|singers?|singing|sung|choir|choral|rap|rapper|a ?cappella|"
    r"vocalist|falsetto)\b", re.IGNORECASE)


def planner_style(style):
    """Build an instrumental style: drop voice requests, add each condition once."""
    parts = [p.strip().rstrip(".").strip() for p in (style or "").split(",")]
    kept = [p for p in parts if p and p.lower() != "instrumental"
            and p.lower() not in _NO_VOCALS
            and (p.lower().startswith("no ") or not _VOCAL_PHRASE.search(p))]
    return ", ".join(["Instrumental", *kept, *_NO_VOCALS])


def _run_tool(name, *args):
    try:
        result = subprocess.run(
            [sys.executable, "-B", str(_SCRIPTS / name), *map(str, args)],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Instrumental tool {name} exceeded 30 seconds") from exc
    if result.returncode:
        raise ValueError(f"Instrumental score preparation failed: {result.stderr.strip()}")


def prepare_instrumental(pipe, request: GenerateRequest, job_dir: Path, plan_kwargs: dict):
    """Return a render request; keep the original plan and transfer evidence.

    Supplied ABC bypasses model composition. Empty/truncated model plans fail
    before synthesis. Upstream prepares the exact instrumental score, validates
    its voices, chooses full/melody mode and adds the no-vocal style conditions.
    """
    style = request.style if request.style is not None else request.tags
    style = planner_style(style)
    planning = dict(plan_kwargs, style=style, lyrics=request.lyrics or _DEFAULT_SECTIONS,
                    cot=request.cot or "full")
    planning.pop("tags", None)
    original = pipe.plan(**planning)
    original_path = job_dir / "original_score.abc"
    if original.abc is not None:
        original_path.write_bytes(original.abc.encode("utf-8"))
    (job_dir / "original_plan.json").write_text(json.dumps({
        "request": original.request.to_dict(), "abc_ids": original.abc_ids,
        "prefix": original.prefix, "timing": original.timing, "truncated": original.truncated,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if original.truncated or not original.abc or not original.abc.strip():
        raise ValueError("Instrumental score is empty or truncated; retry planning before rendering.")

    # Upstream owns parsing, overlap handling, voice transfer and validation.
    # Temporary helper manifests stay private; the final native bundle hashes
    # the flat API artifacts below alongside its own score, tokens and latents.
    with TemporaryDirectory(prefix="instrumental-", dir=job_dir) as temporary:
        root = Path(temporary).resolve()
        conversion, prepared = root / "conversion", root / "prepared"
        _run_tool("instrumentalize.py", original_path.resolve(), "--output", conversion)
        _run_tool("instrumental.py", "prepare", "--abc", conversion / "score.abc",
                  f"--style={style}", "--seed", request.seed,
                  "--empty-lyrics", "--output", prepared)
        data = json.loads((prepared / "request.json").read_text(encoding="utf-8"))
        transfer = json.loads((conversion / "transfer.json").read_text(encoding="utf-8"))
        check = json.loads((prepared / "score-check.json").read_text(encoding="utf-8"))

    (job_dir / "instrumental.json").write_text(json.dumps({
        "instrumental": True, "source": "provided" if request.abc is not None else "generated",
        "api_request": request.model_dump(), "planning_cot": planning["cot"],
        "render_cot": data["cot"], "transfer": transfer, "score_check": check,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return request.model_copy(update={
        "style": data["style"], "tags": None, "lyrics": data["lyrics"],
        "abc": data["abc"], "cot": data["cot"],
    })
