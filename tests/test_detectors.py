import json
import sqlite3
from collections import defaultdict

import pytest

from booksreview import db, detectors
from booksreview.generate import generate


def load(tmp_path, **kw):
    ledger, labels = generate(tmp_path, **kw)
    truth = defaultdict(dict)
    for l in json.loads(labels.read_text())["labels"]:
        truth[l["detector"]][l["evidence_id"]] = l["amount_paise"]
    return db.connect_ro(ledger), truth


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return load(tmp_path_factory.mktemp("clean"))


def test_registry_has_five_detectors():
    assert set(detectors.DETECTORS) == {
        "duplicate_invoice", "gst_mismatch", "unbalanced_voucher", "payment_split", "amount_outlier"}


@pytest.mark.parametrize("name", sorted(detectors.DETECTORS))
def test_recall_and_precision(world, name):
    conn, truth = world
    found = {r["evidence_id"]: r["amount_paise"] for r in detectors.run(conn, name)}
    want = truth[name]
    assert want, f"no seeded labels for {name}"
    assert set(want) <= set(found), f"missed: {set(want) - set(found)}"  # recall 1.0
    assert set(found) <= set(want), f"false positives: {set(found) - set(want)}"  # precision 1.0
    assert found == want  # amounts follow the documented convention


def test_records_have_contract_fields(world):
    conn, _ = world
    for name in detectors.DETECTORS:
        for r in detectors.run(conn, name):
            assert {"detector", "evidence_id", "amount_paise"} <= set(r)
            assert isinstance(r["amount_paise"], int) and r["detector"] == name


def test_limit_applies(world):
    conn, _ = world
    assert len(detectors.run(conn, "amount_outlier", limit=3)) == 3


def test_ledger_is_read_only(world):
    conn, _ = world
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM invoices")


def test_generator_bug_is_visible_when_splits_sit_on_ordinary_vendors(tmp_path):
    """Known problem: oversized split invoices on ordinary vendors are real outliers.
    The detector is right; the generator default avoids the scenario."""
    conn, truth = load(tmp_path, split_on_ordinary_vendors=True)
    extra = {r["evidence_id"] for r in detectors.run(conn, "amount_outlier")} - set(truth["amount_outlier"])
    assert extra, "expected the splitting invoices to be flagged as outliers"
