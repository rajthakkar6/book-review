"""Per-run JSONL trace: every LLM request/response, tool call/result, verdict and the final report.

One file per run under config.TRACE_DIR, one JSON object per line, in order. Meant for manual review.
The API key is never part of any logged object (it lives only in the client).
"""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config


class Trace:
    def __init__(self, path: Path | None):
        self.path, self.seq, self.t0 = path, 0, time.monotonic()
        self.run_id = path.stem if path else None

    @classmethod
    def for_run(cls, label: str = "review") -> "Trace":
        if not config.TRACE_ENABLED:
            return cls(None)
        d = Path(config.TRACE_DIR)
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return cls(d / f"{stamp}-{label}-{uuid.uuid4().hex[:6]}.jsonl")

    def event(self, kind: str, **data):
        if self.path is None:
            return
        self.seq += 1
        rec = {"seq": self.seq, "ts": datetime.now(timezone.utc).isoformat(), "t": round(time.monotonic() - self.t0, 3),
               "kind": kind, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
