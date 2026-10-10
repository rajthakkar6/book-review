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
    e.add_argument("--judge", action="store_true", help="LLM-grade the quality of recommended actions "
                   "(model: BOOKSREVIEW_JUDGE_MODEL)")
    e.add_argument("--price-per-mtok", type=float, default=None, help="blended $/1M tokens, for a cost estimate")
    e.add_argument("--model", default=config.MODEL_NAME)

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
