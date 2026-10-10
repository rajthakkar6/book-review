# Design note: books-review agent

## Problem
Finance teams review ledgers for anomalies (duplicate bills, wrong GST, unbalanced
vouchers, payments split to dodge approval limits, odd amounts). We build an agent
that finds these, explains them, and **never reports anything the ledger did not show it**.

## Division of labour (current design)
Code decides WHAT is reported; the model decides HOW it is worded.
- Every detector record becomes a finding in code (ids, amounts, payment-split clusters). The harness runs any
  detector the model skipped. The model cannot add, drop, regroup or re-price a finding.
- The model runs the detectors (and may inspect with the lookup tools), then writes one explanation per finding
  (keyed by evidence_id, or cluster_id for splits) and an action/rationale per vendor. Missing, malformed or
  unusable text falls back to a template, so a weak model degrades the prose, never coverage or numbers.
- Vendors whose findings span 2+ detector types are computed in code (level: 3+ types = high, 2 = medium;
  order: most types first, ties by vendor_id). The model only words the action for each.
- Why: with model-authored findings, `gpt-4o-mini` mis-added cluster totals, garbled digits, merged clusters and
  mis-ranked vendors (0 of 6 accepted runs). Everything derivable by code is now done by code.

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
| `schemas.py` | `Finding{type, evidence_ids, amount_paise, explanation, explanation_source}`, `VendorRisk`, `Report{status, findings, vendor_risks, summary, partial}`. |
| `llm.py` | `LLM.complete(messages, tools) -> LLMResponse`; OpenAI and Fake implementations. |
| `guards.py` | `Budget`, tool allowlist, output truncation, `check_grounded(findings, seen)`. |
| `agent.py` | LangGraph: model -> tools -> model ... -> assemble (code builds findings; one repair for unusable JSON) -> correlate (vendor notes) -> finish. |
| `baseline.py` | Findings and vendor risks built purely from detector records: the eval baseline and the fallback. |

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
| Model cites an unknown id or amount | Model text is only merged by (type, id); unknown keys are ignored and counted | findings unaffected, `unknown` counted |
| Model skips a detector or stops early | Harness runs missing detectors itself | complete findings |
| Model asks for a non-allowlisted tool | Allowlist rejects | `refused`, no findings |
| Runaway loop | Budget (steps/tools/tokens/wall clock) | `budget_exceeded` + complete code-built findings (`partial`) |
| Unusable final JSON | One repair; then template text | `llm_error` + complete code-built findings (`partial`) |
| LLM API error | Caught at the model node | `llm_error` + code-built findings (`partial`) |
| Detector output cannot be obtained at all | none available | `insufficient_evidence`, no findings |
| Prompt injection in vendor name | Tool output wrapped as untrusted data; allowlist; model cannot author findings | injected instructions not followed, injected ids never appear |
| Model invents a number in its prose | Eval counts numbers in model text not present in detector output | measured, not blocked |
| Detector false positive/negative | recall/precision tests against labels | CI failure |
| Detector output truncated by the tool limit (50) | `total_found` vs `returned` noted in the summary | flagged `[incomplete: ...]` |

## Known issue: outliers vs splitting scenarios
Splitting scenarios need large invoices. If they are placed on ordinary vendors they are genuine
statistical outliers, so `amount_outlier` flags them while the labels do not. The detector is
behaving correctly; the data is unrealistic (a vendor's real scale would include such invoices).
Decision: fix the generator, giving split scenarios to high-value vendors whose normal invoices are
of similar size. Do not weaken the detector to hide generator bugs.
