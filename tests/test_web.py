import json
import time

import pytest
from fastapi.testclient import TestClient

from booksreview.llm import FakeLLM, LLMResponse, ToolCall
from booksreview.web.app import create_app

DETS = ("duplicate_invoice", "gst_mismatch", "unbalanced_voucher", "payment_split", "amount_outlier")


def script():
    calls = [ToolCall(f"c{i}", "run_detector", {"name": n}) for i, n in enumerate(DETS)]
    return FakeLLM([LLMResponse(tool_calls=calls, tokens=5),
                    LLMResponse(content=json.dumps({"explanations": [], "summary": "ignored"}), tokens=5)])


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(llm_factory=script, data_dir=tmp_path / "data")) as c:  # keeps one event loop alive
        yield c


def wait(client, jid, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_page_and_status(client):
    r = client.get("/")
    assert r.status_code == 200 and "Books Review" in r.text
    s = client.get("/api/status").json()
    assert {"model", "api_key_set", "busy"} <= set(s) and s["busy"] is False


def test_generate_lists_ledger_with_labels(client):
    assert client.get("/api/ledgers").json() == []
    r = client.post("/api/generate", json={"scenario": "standard", "seed": 7})
    assert r.status_code == 200
    ls = client.get("/api/ledgers").json()
    assert len(ls) == 1 and ls[0]["has_labels"] and ls[0]["id"] == r.json()["ledger"]
    d = client.post("/api/eval/detectors", json={"ledger": ls[0]["id"]}).json()
    assert set(d) == set(DETS) and all(v["recall"] == 1.0 for v in d.values())


def test_review_job_end_to_end_and_history(client):
    ledger = client.post("/api/generate", json={"scenario": "standard", "seed": 7}).json()["ledger"]
    jid = client.post("/api/review", json={"ledger": ledger}).json()["id"]
    j = wait(client, jid)
    assert j["status"] == "done" and j["result"]["status"] == "ok"
    runs = client.get("/api/runs").json()
    assert len(runs) == 1 and runs[0]["findings"] == 35 and runs[0]["complete"]
    d = client.get(f"/api/runs/{runs[0]['id']}").json()
    assert d["report"]["summary"].startswith("35 findings") and d["stats"]["tool_calls"] == 5
    tr = client.get(f"/api/runs/{runs[0]['id']}/trace").json()
    assert tr[0]["kind"] == "run_start" and all("raw" not in e for e in tr if e["kind"] == "llm_response")
    full = client.get(f"/api/runs/{runs[0]['id']}/trace?full=true").json()
    assert len(full) == len(tr)


def test_second_job_is_rejected_while_one_runs(tmp_path):
    import asyncio

    class Slow(FakeLLM):
        async def complete(self, messages, tools):
            await asyncio.sleep(0.5)
            return await super().complete(messages, tools)
    app = create_app(llm_factory=lambda: Slow(script().script), data_dir=tmp_path / "data")
    with TestClient(app) as c:
        ledger = c.post("/api/generate", json={"scenario": "standard", "seed": 7}).json()["ledger"]
        first = c.post("/api/review", json={"ledger": ledger}).json()
        assert c.post("/api/review", json={"ledger": ledger}).status_code == 409
        assert wait(c, first["id"])["status"] == "done"


@pytest.mark.parametrize("path", ["/api/runs/..%2F..%2Fetc%2Fpasswd", "/api/runs/nope", "/api/evals/nope", "/api/evals/..%2Fx"])
def test_unknown_or_traversal_ids_are_rejected(client, path):
    assert client.get(path).status_code in (400, 404)


def test_ledger_path_cannot_escape_data_dir(client, tmp_path):
    (tmp_path / "ledger.db").write_bytes(b"x")  # exists, but outside data/
    assert client.post("/api/review", json={"ledger": "../ledger.db"}).status_code == 404
    assert client.post("/api/eval/detectors", json={"ledger": "../../etc/passwd"}).status_code == 404


def test_missing_api_key_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    ledger = c.post("/api/generate", json={"scenario": "standard", "seed": 7}).json()["ledger"]
    r = c.post("/api/review", json={"ledger": ledger})
    assert r.status_code == 400 and "OPENAI_API_KEY" in r.json()["detail"]


def test_live_eval_job_and_results_listing(client):
    j = client.post("/api/eval/live", json={"scenarios": ["clean"], "seeds": [7], "runs": 1}).json()
    done = wait(client, j["id"], timeout=60)
    assert done["status"] == "done", done
    evs = client.get("/api/evals").json()
    assert evs and evs[0]["scenarios"] == ["clean"]
    detail = client.get(f"/api/evals/{done['result']['saved_to']}").json()
    assert detail["scenarios"]["clean"]["findings"]["agent"]["overall"]["false_positive_ids"] == 0


def test_bad_request_bodies_are_validated(client):
    assert client.post("/api/eval/live", json={"scenarios": ["standard"], "seeds": [7], "runs": 999}).status_code == 422
    assert client.post("/api/generate", json={"scenario": "weird", "seed": 1}).status_code == 422


# ---- ask
def ask_llm_factory(captured):
    scripts = iter([
        [LLMResponse(tool_calls=[ToolCall("a", "run_detector", {"name": "gst_mismatch"})], tokens=5),
         LLMResponse(content="There are 8 GST mismatches.", tokens=5)],
        [LLMResponse(content="As I said, 8.", tokens=5)],
    ])

    def make():
        llm = FakeLLM(next(scripts))
        captured.append(llm)
        return llm
    return make


def test_ask_conversation_keeps_history_and_reports_tools(tmp_path):
    captured = []
    with TestClient(create_app(llm_factory=ask_llm_factory(captured), data_dir=tmp_path / "data")) as c:
        ledger = c.post("/api/generate", json={"scenario": "standard", "seed": 7}).json()["ledger"]
        j1 = wait(c, c.post("/api/ask", json={"ledger": ledger, "question": "How many GST errors?"}).json()["id"])
        r1 = j1["result"]
        assert j1["status"] == "done" and r1["status"] == "ok" and r1["answer"].startswith("There are 8")
        assert [t["name"] for t in r1["tools"]] == ["run_detector"] and r1["run"]
        j2 = wait(c, c.post("/api/ask", json={"ledger": ledger, "question": "Again?",
                                                 "conversation_id": r1["conversation_id"]}).json()["id"])
        assert j2["result"]["conversation_id"] == r1["conversation_id"]
        sent = captured[1].calls[0]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]  # earlier turn carried over


def test_ask_validates_input_and_failed_turns_are_not_remembered(tmp_path, monkeypatch):
    llms = iter([FakeLLM([RuntimeError("503")]), FakeLLM([LLMResponse(content="fine", tokens=1)])])
    with TestClient(create_app(llm_factory=lambda: next(llms), data_dir=tmp_path / "data")) as c:
        ledger = c.post("/api/generate", json={"scenario": "standard", "seed": 7}).json()["ledger"]
        assert c.post("/api/ask", json={"ledger": ledger, "question": ""}).status_code == 422
        assert c.post("/api/ask", json={"ledger": ledger, "question": "x" * 1001}).status_code == 422
        assert c.post("/api/ask", json={"ledger": "../nope/ledger.db", "question": "hi"}).status_code == 404
        r1 = wait(c, c.post("/api/ask", json={"ledger": ledger, "question": "hello"}).json()["id"])["result"]
        assert r1["status"] == "llm_error"
        r2 = wait(c, c.post("/api/ask", json={"ledger": ledger, "question": "again", "conversation_id": r1["conversation_id"]}).json()["id"])["result"]
        assert r2["status"] == "ok" and r2["conversation_id"] == r1["conversation_id"]


def test_ask_requires_api_key_when_using_the_real_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(create_app(data_dir=tmp_path / "data")) as c:
        ledger = c.post("/api/generate", json={"scenario": "standard", "seed": 7}).json()["ledger"]
        r = c.post("/api/ask", json={"ledger": ledger, "question": "hi"})
        assert r.status_code == 400 and "OPENAI_API_KEY" in r.json()["detail"]
