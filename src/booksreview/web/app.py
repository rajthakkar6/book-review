"""Local web UI: run reviews and evals, browse reports, traces and eval results.

Bound to loopback only (no authentication). Reads the JSONL traces and eval JSON that the CLI also writes,
so everything shown here is also available as files.
"""
import asyncio
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import config
from ..agent import DEFAULT_QUESTION, run_review
from ..ask import ask as ask_ledger

STATIC = Path(__file__).parent / "static"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
SCENARIOS = Literal["standard", "clean", "injection"]


class GenerateReq(BaseModel):
    scenario: SCENARIOS = "standard"
    seed: int = Field(7, ge=0, le=10_000)


class ReviewReq(BaseModel):
    ledger: str
    question: str | None = Field(None, max_length=1000)


class AskReq(BaseModel):
    ledger: str
    question: str = Field(min_length=1, max_length=config.ASK_MAX_QUESTION_CHARS)
    conversation_id: str | None = Field(None, max_length=32)


class DetectorEvalReq(BaseModel):
    ledger: str


class LiveEvalReq(BaseModel):
    scenarios: list[SCENARIOS] = ["standard"]
    seeds: list[int] = Field([7], min_length=1, max_length=8)
    runs: int = Field(3, ge=1, le=10)
    judge: bool = False


def _evals():
    root = Path(__file__).resolve().parents[3]  # evals/ lives next to src/, outside the package
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evals import run_evals
    return run_evals


def _read_events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def create_app(llm_factory=None, logs_dir: Path | None = None, data_dir: Path | None = None) -> FastAPI:
    """llm_factory: () -> LLM. Defaults to the configured OpenAI model; tests inject a FakeLLM."""
    logs = Path(logs_dir or config.TRACE_DIR)
    data = Path(data_dir or config.DATA_DIR)
    default_llm = llm_factory is None
    jobs: dict[str, dict] = {}
    convos: dict[str, dict] = {}  # conversation id -> {ledger, messages, allowed numbers}; in memory only
    lock = threading.Lock()
    app = FastAPI(title="Books review", docs_url=None, redoc_url=None)

    def make_llm(model: str | None = None):
        if default_llm:
            if not os.environ.get("OPENAI_API_KEY"):
                raise HTTPException(400, "OPENAI_API_KEY is not set (put it in .env and restart the server).")
            from ..llm import OpenAILLM
            return OpenAILLM(model or config.MODEL_NAME)
        return llm_factory()

    def resolve_ledger(ledger_id: str) -> Path:
        p = (data / ledger_id).resolve()
        if not p.is_relative_to(data.resolve()) or p.name != "ledger.db" or not p.is_file():
            raise HTTPException(404, "unknown ledger")
        return p

    def trace_path(run_id: str) -> Path:
        if not SAFE_ID.match(run_id):
            raise HTTPException(400, "bad id")
        p = logs / f"{run_id}.jsonl"
        if not p.is_file():
            raise HTTPException(404, "unknown run")
        return p

    def eval_path(eval_id: str) -> Path:
        if not SAFE_ID.match(eval_id):
            raise HTTPException(400, "bad id")
        p = logs / f"{eval_id}.json"
        if not p.is_file():
            raise HTTPException(404, "unknown eval")
        return p

    def start_job(kind: str, work):
        """work: sync callable (run in a thread) or coroutine function. One job at a time."""
        with lock:
            if any(j["status"] == "running" for j in jobs.values()):
                raise HTTPException(409, "another job is still running")
            jid = uuid.uuid4().hex[:8]
            jobs[jid] = {"id": jid, "kind": kind, "status": "running", "started": time.time(), "result": None, "error": None}

        def finish(result=None, error=None):
            jobs[jid].update(status="error" if error else "done", result=result, error=error, finished=time.time())

        if asyncio.iscoroutinefunction(work):
            async def runner():
                try:
                    finish(result=await work())
                except Exception as e:  # surfaced to the UI, never crashes the server
                    finish(error=f"{type(e).__name__}: {str(e)[:300]}")
            asyncio.get_running_loop().create_task(runner())
        else:
            def runner_sync():
                try:
                    finish(result=work())
                except Exception as e:
                    finish(error=f"{type(e).__name__}: {str(e)[:300]}")
            threading.Thread(target=runner_sync, daemon=True).start()
        return jobs[jid]

    # ---- pages and status
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {"model": config.MODEL_NAME, "judge_model": config.JUDGE_MODEL, "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
                "reasoning_effort": config.REASONING_EFFORT or None, "tracing": config.TRACE_ENABLED,
                "busy": any(j["status"] == "running" for j in jobs.values())}

    # ---- ledgers
    @app.get("/api/ledgers")
    def ledgers():
        found = [p for p in [data / "ledger.db", *sorted(data.glob("*/ledger.db"))] if p.is_file()]
        return [{"id": str(p.relative_to(data)), "has_labels": (p.parent / "labels.json").is_file(),
                 "modified": p.stat().st_mtime} for p in found]

    @app.post("/api/generate")
    def generate(req: GenerateReq):
        ev = _evals()
        data.mkdir(parents=True, exist_ok=True)
        ledger, _ = ev.prepare(req.scenario, req.seed, data)
        return {"ledger": str(ledger.relative_to(data))}

    # ---- jobs
    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        if jid not in jobs:
            raise HTTPException(404, "unknown job")
        return jobs[jid]

    @app.post("/api/review")
    async def review(req: ReviewReq):
        ledger = resolve_ledger(req.ledger)
        llm = make_llm()

        async def work():
            report, stats = await run_review(llm, ledger, req.question or DEFAULT_QUESTION)
            return {"status": report.status, "run": Path(stats["trace"]).stem if stats.get("trace") else None, "stats": stats}
        return start_job("review", work)

    @app.post("/api/ask")
    async def ask_endpoint(req: AskReq):
        ledger = resolve_ledger(req.ledger)
        llm = make_llm()
        question = req.question.strip()
        if not question:
            raise HTTPException(422, "empty question")
        convo = convos.get(req.conversation_id or "")
        if convo is None or convo["ledger"] != req.ledger:  # unknown id or a different ledger: start fresh
            cid = uuid.uuid4().hex[:12]
            convo = convos[cid] = {"id": cid, "ledger": req.ledger, "messages": [], "allowed": set()}
            for old in list(convos)[:-20]:  # keep the 20 most recent conversations
                convos.pop(old, None)

        async def work():
            r = await ask_ledger(llm, ledger, question, convo["messages"], convo["allowed"])
            if r.status == "ok":  # failed turns don't pollute the conversation
                convo["messages"] += [{"role": "user", "content": question}, {"role": "assistant", "content": r.answer}]
            return {"conversation_id": convo["id"], "status": r.status, "answer": r.answer, "tools": r.tools,
                    "unsupported_numbers": r.unsupported_numbers, "stats": r.stats,
                    "run": Path(r.stats["trace"]).stem if r.stats.get("trace") else None}
        return start_job("ask", work)

    @app.post("/api/eval/detectors")
    def eval_detectors(req: DetectorEvalReq):
        ledger = resolve_ledger(req.ledger)
        labels = ledger.parent / "labels.json"
        if not labels.is_file():
            raise HTTPException(400, "this ledger has no labels.json")
        return _evals().detector_eval(ledger, labels)

    @app.post("/api/eval/live")
    def eval_live(req: LiveEvalReq):
        make_llm()  # fail fast when the key is missing
        ev = _evals()

        def work():
            judge = (lambda: make_llm(config.JUDGE_MODEL)) if req.judge else None
            r = ev.live_eval(lambda: make_llm(), req.scenarios, req.seeds, req.runs, judge_factory=judge)
            return {"saved_to": Path(r["saved_to"]).stem}
        return start_job("live_eval", work)

    # ---- history: reviews
    @app.get("/api/runs")
    def runs():
        out = []
        for p in sorted(logs.glob("*-review-*.jsonl"), reverse=True)[:200]:
            ev = _read_events(p)
            start = next((e for e in ev if e["kind"] == "run_start"), {})
            end = next((e for e in reversed(ev) if e["kind"] == "run_end"), None)
            rep, st = (end or {}).get("report", {}), (end or {}).get("stats", {})
            out.append({"id": p.stem, "ts": start.get("ts"), "model": start.get("model"), "complete": end is not None,
                        "status": rep.get("status"), "partial": rep.get("partial"), "findings": len(rep.get("findings", [])),
                        "vendor_risks": len(rep.get("vendor_risks", [])), "tokens": st.get("tokens"),
                        "seconds": st.get("seconds")})
        return out

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        ev = _read_events(trace_path(run_id))
        end = next((e for e in reversed(ev) if e["kind"] == "run_end"), None)
        if end is None:
            raise HTTPException(404, "run did not finish")
        return {"id": run_id, "start": next((e for e in ev if e["kind"] == "run_start"), {}),
                "report": end["report"], "stats": end["stats"]}

    @app.get("/api/runs/{run_id}/trace")
    def run_trace(run_id: str, full: bool = False):
        ev = _read_events(trace_path(run_id))
        if not full:  # raw provider payloads are large; the request/response fields already hold the content
            ev = [{k: v for k, v in e.items() if k != "raw" or e["kind"] != "llm_response"} for e in ev]
        return ev

    # ---- history: evals
    @app.get("/api/evals")
    def evals():
        out = []
        for p in sorted(logs.glob("eval-*.json"), reverse=True)[:100]:
            try:
                j = json.loads(p.read_text())
                out.append({"id": p.stem, "config": j.get("config"), "scenarios": list(j.get("scenarios", {}))})
            except ValueError:
                continue
        return out

    @app.get("/api/evals/{eval_id}")
    def eval_detail(eval_id: str):
        return json.loads(eval_path(eval_id).read_text())

    return app
