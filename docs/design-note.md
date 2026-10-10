# Design note: books-review agent

## Problem
Finance teams review ledgers for anomalies (duplicate bills, wrong GST, unbalanced
vouchers, payments split to dodge approval limits, odd amounts). We build an agent
that finds these, explains them, and **never reports anything the ledger did not show it**.

## Constraints
- Read-only: the agent can never modify the ledger (enforced by SQLite `mode=ro` + `query_only`).
- No arbitrary SQL for the model. It gets five narrow tools; `limit` is capped at 50.
- Money is integer paise everywhere. No floats in storage or comparisons.
- Detection is deterministic code. The LLM orchestrates, explains and summarises; it does not detect.
- Every finding must be grounded: each cited `evidence_id` and `amount_paise` must have
  appeared in a tool result during the same run.
- Bounded: steps, tool calls, tokens and wall clock are all capped.
- Testable offline: a scripted FakeLLM stands in for OpenAI.

## Components and contracts
| Part | Contract |
|---|---|
| `db.py` | Only module that opens the ledger. Read-only connection. Returns plain dicts. |
| `generate.py` | Writes `ledger.db` and a separate `labels.json` (ground truth). Agent code never reads labels. |
| `detectors.py` | `run(conn, name, limit) -> list[Record]`; `Record = {detector, evidence_id, amount_paise, detail}`. |
| `mcp_server.py` | Tools `list_detectors`, `run_detector`, `get_invoice`, `get_voucher`, `vendor_summary`. All read-only. |
| `schemas.py` | `Finding{type, evidence_ids, amount_paise, explanation}`, `Report{status, findings, summary}`. |
| `llm.py` | `LLM.complete(messages, tools) -> LLMResponse`; OpenAI and Fake implementations. |
| `guards.py` | `Budget`, tool allowlist, output truncation, `check_grounded(findings, seen)`. |
| `agent.py` | LangGraph: model -> tools -> model ... -> validate -> (repair once) -> finish. |

Evidence conventions (`amount_paise`): duplicate / outlier = invoice total; GST mismatch =
absolute GST error; unbalanced voucher = absolute debit-credit gap; payment split = that payment's amount.

## Detector definitions
- **duplicate_invoice**: same vendor and same normalised invoice number (case/punctuation stripped)
  or same vendor+date+total. The later invoice(s) are flagged.
- **gst_mismatch**: `gst_paise` differs from `round(taxable * rate / 100)` by more than 1 paisa.
- **unbalanced_voucher**: sum(debit) != sum(credit) for a voucher.
- **payment_split**: payments to one vendor within a window, each below the approval limit and at
  least a minimum fraction of it, whose sum reaches the limit. Every payment in the cluster is flagged.
- **amount_outlier**: invoice total above vendor median by more than k scaled MADs, with a minimum
  history size per vendor.

## End statuses
`ok`, `refused`, `budget_exceeded`, `insufficient_evidence`, `llm_error`.

## Failure modes
| Failure | Defence | Result |
|---|---|---|
| Model invents an evidence id | groundedness check; one repair retry | repaired or `insufficient_evidence` |
| Model cites wrong amount | same check | same |
| Model calls non-allowlisted tool | allowlist rejects, error returned to model | continues or `refused` |
| Runaway loop | Budget (steps/tools/tokens/wall clock) | `budget_exceeded` |
| Malformed JSON final answer | parse failure triggers one repair | repaired or `llm_error` |
| Prompt injection in vendor name | tool output is data, wrapped and truncated; tool allowlist; groundedness | injected instructions not followed |
| LLM API error | caught at the model node | `llm_error` |
| Detector false positive/negative | recall/precision tests against labels | CI failure |
| Generator vs detector disagree | treated as an eval question: decide which side is wrong | see below |

## Known issue: outliers vs splitting scenarios
Splitting scenarios need large invoices. If they are placed on ordinary vendors they are genuine
statistical outliers, so `amount_outlier` flags them while the labels do not. The detector is
behaving correctly; the data is unrealistic (a vendor's real scale would include such invoices).
Decision: fix the generator, giving split scenarios to high-value vendors whose normal invoices are
of similar size. Do not weaken the detector to hide generator bugs.
