"""Entry point: booksreview generate | review | eval."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from . import config


def _llm(model: str):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set; live runs need it.")
    from .llm import OpenAILLM
    return OpenAILLM(model)


def _evals():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evals import run_evals  # lives outside the package
    return run_evals


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="booksreview")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create a synthetic ledger and ground-truth labels")
    g.add_argument("--out", default=str(config.DATA_DIR))
    g.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("review", help="run the agent against a live model")
    r.add_argument("--ledger", default=str(config.LEDGER_PATH))
    r.add_argument("--model", default=config.MODEL_NAME)
    r.add_argument("--question", default=None)

    e = sub.add_parser("eval", help="detector recall/precision; add --live for agent metrics")
    e.add_argument("--data", default=str(config.DATA_DIR))
    e.add_argument("--live", action="store_true")
    e.add_argument("--runs", type=int, default=5, help="runs per scenario per seed")
    e.add_argument("--seeds", default="7", help="comma-separated generator seeds")
    e.add_argument("--scenarios", default="standard,clean,injection")
    e.add_argument("--judge", action="store_true", help="blind LLM comparison of the model's wording vs the template's "
                   "(model: BOOKSREVIEW_JUDGE_MODEL)")
    e.add_argument("--price-per-mtok", type=float, default=None, help="blended $/1M tokens, for a cost estimate")
    e.add_argument("--model", default=config.MODEL_NAME)

    w = sub.add_parser("serve", help="local web UI (run reviews/evals, browse reports and traces)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)

    if a.cmd == "generate":
        from .generate import generate
        ledger, labels = generate(a.out, seed=a.seed)
        print(f"ledger: {ledger}\nlabels: {labels}")
    elif a.cmd == "review":
        from .agent import DEFAULT_QUESTION, run_review
        report, stats = asyncio.run(run_review(_llm(a.model), a.ledger, a.question or DEFAULT_QUESTION))
        print(json.dumps({"report": report.model_dump(), "stats": stats}, indent=2))
        return 0 if report.status == "ok" else 1
    elif a.cmd == "serve":
        if a.host not in ("127.0.0.1", "localhost", "::1"):
            sys.exit("The UI has no authentication and can spend API tokens: it only binds to loopback.")
        try:
            import uvicorn
            from .web.app import create_app
        except ImportError:
            sys.exit('Web UI needs extra packages: pip install -e ".[web]"')
        print(f"Books review UI: http://{a.host}:{a.port}  (Ctrl+C to stop)")
        uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
    elif a.cmd == "eval":
        ev, d = _evals(), Path(a.data)
        out = {"detectors": ev.detector_eval(d / "ledger.db", d / "labels.json")}
        if a.live:
            out["live"] = ev.live_eval(lambda: _llm(a.model), a.scenarios.split(","),
                                       [int(x) for x in a.seeds.split(",")], a.runs,
                                       price_per_mtok=a.price_per_mtok,
                                       judge_factory=(lambda: _llm(config.JUDGE_MODEL)) if a.judge else None)
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
