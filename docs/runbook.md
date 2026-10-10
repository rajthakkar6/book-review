# Runbook: what to check when it breaks

## First: read the end status
`booksreview review` prints `report.status` and `stats` (steps, tool_calls, tokens, seconds, repairs).

| Status | Meaning | Check |
|---|---|---|
| `ok` | model text accepted | `stats.explanations` (model vs template coverage) and `stats.notes` show how much prose came from the model |
| `llm_error` | provider failed, or no usable JSON twice | API key, quota, model name, `BOOKSREVIEW_REASONING_EFFORT` support; `report.partial` is true and findings are still complete (code-written text) |
| `budget_exceeded` | a limit in `config.py` was hit; `summary` says which | model looping? findings are still complete (`partial`); raise a limit only after reading the transcript |
| `refused` | model asked for a tool outside the allowlist; nothing is reported | possible prompt injection in ledger text; inspect vendor/invoice text |
| `insufficient_evidence` | no detector output could be obtained at all | ledger path, MCP server, trace `harness_ran_detector` events |
| findings look wrong | detector issue, not agent | `booksreview eval`; recall/precision should be 1.0 on seeded data |

## Common problems
- **`unable to open database file`**: ledger path wrong, or file missing. Run `booksreview generate`. In Docker the volume must be mounted at `/app/data`.
- **`attempt to write a readonly database`**: expected. The agent connection is read-only by design; only `generate` writes.
- **Detector recall dropped**: thresholds in `config.py` changed, or the generator changed. Compare `labels.json` to detector output.
- **`reasoning_effort` 400 error**: non-reasoning models (e.g. gpt-4o-mini) reject it; leave `BOOKSREVIEW_REASONING_EFFORT` unset. `gpt-6-luna` needs it set to `none`.
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
