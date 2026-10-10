import json

import pytest
from mcp import Client

from booksreview.generate import generate
from booksreview.mcp_server import build_server

TOOLS = {"list_detectors", "run_detector", "get_invoice", "get_voucher", "vendor_summary"}


@pytest.fixture(scope="module")
def ledger(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("mcp"))[0]


async def call(ledger, name, args=None):
    async with Client(build_server(ledger)) as c:
        return await c.call_tool(name, args or {})


def _json_of(r):
    return json.loads(r.content[0].text)


async def test_exactly_five_read_only_tools(ledger):
    async with Client(build_server(ledger)) as c:
        tools = (await c.list_tools()).tools
    assert {t.name for t in tools} == TOOLS  # no SQL tool
    assert all(t.annotations.read_only_hint is True and t.annotations.destructive_hint is False for t in tools)


async def test_list_detectors(ledger):
    r = _json_of(await call(ledger, "list_detectors"))
    assert len(r["detectors"]) == 5


async def test_run_detector_returns_evidence(ledger):
    r = _json_of(await call(ledger, "run_detector", {"name": "unbalanced_voucher"}))
    assert r["total_found"] == 6 and r["records"][0]["evidence_id"].startswith("VCH-")


async def test_limit_is_capped(ledger, monkeypatch):
    from booksreview import config
    assert config.MAX_LIMIT == 50
    monkeypatch.setattr(config, "MAX_LIMIT", 3)  # no detector yields >50 rows, so shrink the cap to exercise it
    r = _json_of(await call(ledger, "run_detector", {"name": "gst_mismatch", "limit": 10_000}))
    assert r["returned"] == 3 and r["total_found"] == 8 and len(r["records"]) == 3
    r = _json_of(await call(ledger, "run_detector", {"name": "gst_mismatch", "limit": 0}))
    assert r["returned"] == 1  # floor of 1


async def test_lookups(ledger):
    assert _json_of(await call(ledger, "get_invoice", {"invoice_id": "INV-0001"}))["vendor_id"] == "V001"
    v = _json_of(await call(ledger, "get_voucher", {"voucher_id": "VCH-0001"}))
    assert v["debit_paise"] == v["credit_paise"] and len(v["lines"]) == 3
    assert _json_of(await call(ledger, "vendor_summary", {"vendor_id": "V001"}))["invoice_count"] >= 30


@pytest.mark.parametrize("tool,args", [
    ("run_detector", {"name": "drop_table"}),
    ("get_invoice", {"invoice_id": "x'; DROP TABLE invoices;--"}),
    ("get_voucher", {"voucher_id": "nope"}),
    ("vendor_summary", {"vendor_id": "nope"}),
])
async def test_bad_input_is_an_error_not_a_crash(ledger, tool, args):
    r = await call(ledger, tool, args)
    assert r.is_error


async def test_injection_string_does_not_alter_ledger(ledger):
    await call(ledger, "get_invoice", {"invoice_id": "x'; DROP TABLE invoices;--"})
    assert (await call(ledger, "get_invoice", {"invoice_id": "INV-0001"})).is_error is False
