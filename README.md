# books-review-agent

A read-only ledger review agent. Deterministic detectors find anomalies; an LLM orchestrates them
through a narrow MCP tool layer and explains the results. Every cited id and amount must have been
returned by a tool in the same run, or the answer is rejected.

See [docs/design-note.md](docs/design-note.md) for the design and [docs/runbook.md](docs/runbook.md) for operations.

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/booksreview generate          # data/ledger.db + data/labels.json
.venv/bin/booksreview eval              # detector recall/precision (no API key)
export OPENAI_API_KEY=...               # live runs only
.venv/bin/booksreview review            # run the agent
.venv/bin/booksreview eval --live --runs 3
.venv/bin/python -m pytest              # all tests, offline
```

## Layout
| Path | Role |
|---|---|
| `src/booksreview/db.py` | only code that opens the ledger (read-only) |
| `generate.py`, `detectors.py` | synthetic ledger with seeded anomalies; five detectors |
| `mcp_server.py` | five read-only tools, limit capped at 50, no SQL tool |
| `guards.py`, `agent.py`, `llm.py`, `schemas.py` | budget, groundedness, LangGraph agent, LLM interface |
| `evals/run_evals.py` | detector recall/precision; live groundedness, tool calls, tokens, latency |

## Notes
- Money is integer paise throughout.
- Configure in `.env` (see `.env.example`): `OPENAI_API_KEY`, `BOOKSREVIEW_MODEL`. Live-tested with `gpt-4o-mini` and `gpt-6-luna` (the latter needs `BOOKSREVIEW_REASONING_EFFORT=none`; non-reasoning models must leave it unset). For non-OpenAI endpoints set `OPENAI_BASE_URL`.
- Design: code builds every finding from detector records; the model only writes the wording. See the design note.
- Every run writes a full JSONL trace to `logs/` (disable with `BOOKSREVIEW_TRACE=0`). Optional LangSmith: `LANGSMITH_TRACING=true`.
- End statuses: `ok`, `refused`, `budget_exceeded`, `insufficient_evidence`, `llm_error`. The CLI exits non-zero unless `ok`.
- Docker: `docker compose --profile tools run --rm generate`, then `review` or `eval`. The ledger is mounted read-only for the agent.
