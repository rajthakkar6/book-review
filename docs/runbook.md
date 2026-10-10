# Runbook: what to check when it breaks

## First: read the end status
`booksreview review` prints `report.status` and `stats` (steps, tool_calls, tokens, seconds, repairs).

| Status | Meaning | Check |
|---|---|---|
| `llm_error` | provider call failed, or the final answer was not valid JSON twice | API key, quota, model name in `config.MODEL_NAME`, provider status; read `report.summary` |
| `budget_exceeded` | a limit in `config.py` was hit; `summary` says which | Model looping? Raise the limit only after reading the transcript |
| `insufficient_evidence` | answer cited ids/amounts no tool returned, even after one repair | Model hallucinating; try a stronger model; check truncation (`truncated_items`) is not hiding records |
| `refused` | model asked for a tool outside the allowlist | Possible prompt injection in ledger text; inspect vendor/invoice text for instructions |
| `ok` but findings look wrong | detector issue, not agent | Run `booksreview eval`; recall/precision should be 1.0 on seeded data |

## Common problems
- **`unable to open database file`**: ledger path wrong, or file missing. Run `booksreview generate`. In Docker the volume must be mounted at `/app/data`.
- **`attempt to write a readonly database`**: expected. The agent connection is read-only by design; only `generate` writes.
- **Detector recall dropped**: thresholds in `config.py` changed, or the generator changed. Compare `labels.json` to detector output.
- **Extra `amount_outlier` hits**: scenario invoices sit on ordinary vendors. Fix the generator, not the detector (see design note).
- **Tools fail after an `mcp` upgrade**: pinned to `mcp>=2.3,<3`. Check `MCPServer`/`Client` signatures.
- **Slow or costly runs**: check `stats.tokens` and `tool_calls`; lower `MAX_TOOL_OUTPUT_CHARS` or `limit`.

## Quick diagnostics
```bash
python -m pytest -q                 # offline; should be green
booksreview eval                    # detector health
booksreview eval --live --runs 3    # agent health (needs key, costs tokens)
```

## Not yet in place
Tracing (Langfuse / OpenTelemetry), Postgres backend, cloud deployment.
