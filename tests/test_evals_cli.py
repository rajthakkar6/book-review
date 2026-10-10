import json

from booksreview import db, detectors
from booksreview.baseline import baseline_findings, baseline_vendor_risks, finding_key
from booksreview.cli import main
from booksreview.llm import FakeLLM, LLMResponse, ToolCall
from evals import run_evals as ev


def test_cli_generate_then_detector_eval(tmp_path, capsys):
    assert main(["generate", "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["eval", "--data", str(tmp_path)]) == 0
    d = json.loads(capsys.readouterr().out)["detectors"]
    assert all(v["recall"] == 1.0 and v["precision"] == 1.0 for v in d.values())


def test_review_without_key_exits(tmp_path, monkeypatch):
    import pytest
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["review", "--ledger", str(tmp_path / "x.db")])


def test_scorer_baseline_is_perfect_and_detects_gaps(tmp_path):
    ledger, truth = ev.prepare("standard", 7, tmp_path)
    full = ev.aggregate([ev.baseline_score(ledger, truth)])
    assert full["overall"]["recall"] == 1.0 and full["overall"]["precision"] == 1.0
    assert full["overall"]["amount_accuracy"] == 1.0
    # drop half of the findings -> recall must fall; add a bogus one -> precision must fall
    conn = db.connect_ro(ledger)
    fs = baseline_findings({n: detectors.run(conn, n) for n in detectors.DETECTORS})
    half = ev.aggregate([ev.score_findings(fs[::2], truth)])["overall"]
    assert 0.3 < half["recall"] < 0.7 and half["precision"] == 1.0
    bad = fs[0].model_copy(update={"evidence_ids": ["INV-9999"]})
    assert ev.aggregate([ev.score_findings(fs + [bad], truth)])["overall"]["precision"] < 1.0
    wrong_amt = fs[0].model_copy(update={"amount_paise": fs[0].amount_paise + 1})
    assert ev.aggregate([ev.score_findings([wrong_amt], truth)])["overall"]["amount_accuracy"] == 0.0


def test_clean_ledger_has_no_detector_hits(tmp_path):
    ledger, truth = ev.prepare("clean", 7, tmp_path)
    conn = db.connect_ro(ledger)
    assert truth == {} and all(detectors.run(conn, n) == [] for n in detectors.DETECTORS)


def test_payment_split_findings_are_per_cluster(tmp_path):
    ledger, _ = ev.prepare("standard", 7, tmp_path)
    fs = baseline_findings({"payment_split": detectors.run(db.connect_ro(ledger), "payment_split")})
    assert len(fs) == 5 and all(len(f.evidence_ids) == 2 for f in fs)


def test_live_eval_machinery_with_fake_llm(tmp_path):
    """Scripted agent that runs all detectors and explains every finding."""
    import re

    def explain(messages):
        recs = {}
        for m in messages:
            if m["role"] == "tool":
                body = json.loads(re.search(r"<tool_output[^>]*>(.*)</tool_output>", m["content"], re.S).group(1))
                recs[body["detector"]] = {r["evidence_id"]: r for r in body["records"]}
        fs = baseline_findings(recs)
        return LLMResponse(content=json.dumps({"explanations": [
            {"type": f.type, "id": finding_key(f, recs)[1], "explanation": f"Model: {f.evidence_ids[0]}"} for f in fs],
            "summary": "ok"}), tokens=7)

    def notes(messages):
        vs = re.findall(r"^- (V\d+) \[", messages[-1]["content"], re.M)
        return LLMResponse(content=json.dumps({"vendor_notes": [
            {"vendor_id": v, "action": f"Hold {v}", "rationale": "x"} for v in vs]}), tokens=3)

    def make():
        calls = [ToolCall(f"c{i}", "run_detector", {"name": n}) for i, n in enumerate(detectors.DETECTORS)]
        return FakeLLM([LLMResponse(tool_calls=calls, tokens=5), explain, notes])
    r = ev.live_eval(make, scenarios=("standard", "clean"), seeds=(7,), runs=2, workdir=tmp_path)
    s, c = r["scenarios"]["standard"], r["scenarios"]["clean"]
    assert s["findings"]["agent"]["overall"]["recall"] == 1.0 and s["rates"]["completed_ok"] == 1.0
    assert s["cost"]["mean_tool_calls"] == 5 and s["findings"]["baseline"]["overall"]["recall"] == 1.0
    assert s["explanations"]["model_coverage"] == 1.0 and s["explanations"]["partial_runs"] == 0
    assert s["vendor_risks"]["agent"]["vendor_recall"] == 1.0 and s["vendor_risks"]["agent"]["order_accuracy"] == 1.0
    assert c["findings"]["agent"]["overall"]["false_positive_ids"] == 0


def test_budget_stop_returns_complete_code_built_findings(tmp_path):
    import asyncio

    from booksreview.agent import run_review
    from booksreview.guards import Budget
    ledger, truth = ev.prepare("standard", 7, tmp_path)
    llm = FakeLLM([LLMResponse(tool_calls=[ToolCall("a", "run_detector", {"name": "gst_mismatch"})], tokens=1),
                   LLMResponse(tool_calls=[ToolCall("b", "list_detectors", {})], tokens=1)])
    report, _ = asyncio.run(run_review(llm, ledger, budget=Budget(max_tool_calls=1)))
    assert report.status == "budget_exceeded" and report.partial and len(report.findings) == 35
    assert ev.aggregate([ev.score_findings(report.findings, truth)])["overall"]["recall"] == 1.0


def test_compound_vendors_are_seeded_and_baseline_scores_perfectly(tmp_path):
    ledger, truth = ev.prepare("standard", 7, tmp_path)
    vt = ev.vendor_truth(ledger, truth)
    exp = ev.expected_risks(vt)
    assert len(exp) >= 3 and exp[0][1] == "high"  # three planted vendors (plus any coincidences)
    conn = db.connect_ro(ledger)
    recs = {n: detectors.run(conn, n) for n in detectors.DETECTORS}
    agg = ev.aggregate_vr([ev.score_vendor_risks(baseline_vendor_risks(recs), vt)])
    assert all(v == 1.0 for v in agg.values())
    assert ev.aggregate_vr([ev.score_vendor_risks([], vt)])["vendor_recall"] == 0.0


def test_number_support_flags_invented_numbers_in_model_text_only(tmp_path):
    from booksreview.schemas import Finding, Report
    ledger, _ = ev.prepare("standard", 7, tmp_path)
    allowed = ev.allowed_numbers(ledger)
    r = detectors.run(db.connect_ro(ledger), "gst_mismatch")[0]
    base = dict(type="gst_mismatch", evidence_ids=[r["evidence_id"]], amount_paise=r["amount_paise"])
    ok = Finding(**base, explanation=f"GST off by {r['amount_paise']:,} paise on {r['evidence_id']} (V005).",
                 explanation_source="model")
    bad = Finding(**base, explanation="GST off by 987654321 paise.", explanation_source="model")
    template = Finding(**base, explanation="GST off by 987654321 paise.", explanation_source="template")
    rep = lambda f: ev.number_support(Report(status="ok", findings=[f]), allowed)  # noqa: E731
    assert rep(ok)["unsupported"] == 0 and rep(ok)["checked"] == 1
    assert rep(bad)["unsupported"] == 1
    assert rep(template)["checked"] == 0  # template text is code-written, so it is not graded


def test_injection_text_reaches_the_model_in_the_eval(tmp_path):
    def make():
        return FakeLLM([LLMResponse(tool_calls=[ToolCall("c", "run_detector", {"name": "gst_mismatch"})], tokens=1),
                        LLMResponse(content='{"findings": [], "summary": "x"}', tokens=1)])
    r = ev.live_eval(make, scenarios=("injection",), seeds=(7,), runs=1, workdir=tmp_path)
    assert r["scenarios"]["injection"]["injection"] == {"runs_where_text_reached_model": 1, "runs_where_followed": 0}
