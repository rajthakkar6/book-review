"""Synthetic ledger with seeded anomalies. Ground truth goes to a separate labels file.

All money is integer paise. This is the only module that writes to the ledger.
"""
import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE vendors (vendor_id TEXT PRIMARY KEY, name TEXT NOT NULL, gstin TEXT);
CREATE TABLE invoices (
  invoice_id TEXT PRIMARY KEY, vendor_id TEXT NOT NULL REFERENCES vendors(vendor_id),
  invoice_no TEXT NOT NULL, invoice_date TEXT NOT NULL,
  taxable_paise INTEGER NOT NULL, gst_rate_pct INTEGER NOT NULL,
  gst_paise INTEGER NOT NULL, total_paise INTEGER NOT NULL, voucher_id TEXT NOT NULL);
CREATE TABLE payments (
  payment_id TEXT PRIMARY KEY, vendor_id TEXT NOT NULL REFERENCES vendors(vendor_id),
  invoice_id TEXT, pay_date TEXT NOT NULL, amount_paise INTEGER NOT NULL);
CREATE TABLE voucher_lines (
  voucher_id TEXT NOT NULL, line_no INTEGER NOT NULL, account TEXT NOT NULL,
  debit_paise INTEGER NOT NULL DEFAULT 0, credit_paise INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (voucher_id, line_no));
"""

N_ORDINARY, N_LARGE, INV_PER_VENDOR = 26, 4, 30
COUNTS = dict(duplicate=8, gst=8, unbalanced=6, outlier=8, split=5)
RUPEE = 100
COMPOUND_PLAN = [["gst", "outlier", "duplicate"], ["gst", "unbalanced"],
                 ["duplicate", "outlier", "unbalanced", "gst"]]


def _gst(taxable: int, rate: int) -> int:
    return (taxable * rate + 50) // 100  # half-up, integer only


def generate(out_dir: Path | str = config.DATA_DIR, seed: int = 7,
             split_on_ordinary_vendors: bool = False, counts: dict | None = None) -> tuple[Path, Path]:
    """Build ledger.db and labels.json in out_dir.

    split_on_ordinary_vendors=True reproduces the known generator bug: splitting
    scenarios are inflated invoices on ordinary vendors, so they are real outliers.
    """
    counts = {**COUNTS, **(counts or {})}  # e.g. all zeros for a clean ledger
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger, labels_path = out / "ledger.db", out / "labels.json"
    ledger.unlink(missing_ok=True)
    limit = config.APPROVAL_LIMIT_PAISE

    vendors, large = [], set()
    for i in range(N_ORDINARY + N_LARGE):
        vid = f"V{i+1:03d}"
        is_large = i >= N_ORDINARY
        if is_large:
            large.add(vid)
        base = rng.randint(600_000, 700_000) * RUPEE if is_large else rng.randint(20_000, 150_000) * RUPEE
        vendors.append(dict(vendor_id=vid, name=f"Vendor {i+1} Traders", gstin=f"27AAAAA{i:04d}A1Z5", base=base))

    # Ordinary invoices. Per-vendor date spacing >= 9 days so normal payments never cluster.
    invoices, n = [], 0
    for v in vendors:
        d = date(2025, 4, 1) + timedelta(days=rng.randint(0, 5))
        for k in range(INV_PER_VENDOR):
            n += 1
            rate = rng.choice([5, 12, 18])
            lo, hi = (0.9, 1.1)
            taxable = int(v["base"] * rng.uniform(lo, hi))
            if v["vendor_id"] in large:  # approved single payments, always >= limit
                while taxable + _gst(taxable, rate) < limit:
                    taxable = int(taxable * 1.05)
            gst = _gst(taxable, rate)
            invoices.append(dict(
                invoice_id=f"INV-{n:04d}", vendor_id=v["vendor_id"], invoice_no=f"{v['vendor_id']}/{k+1:04d}",
                invoice_date=d, taxable_paise=taxable, gst_rate_pct=rate, gst_paise=gst,
                total_paise=taxable + gst, voucher_id=f"VCH-{n:04d}", pay_split=None, voucher_gap=0))
            d += timedelta(days=12 + rng.randint(0, 3))
    originals = list(invoices)
    labels, used = [], set()

    def pick(pool):
        c = [x for x in pool if x["invoice_id"] not in used]
        x = rng.choice(c)
        used.add(x["invoice_id"])
        return x

    ordinary = [x for x in originals if x["vendor_id"] not in large]
    largeinv = [x for x in originals if x["vendor_id"] in large]

    dups = []

    def mk_gst(x):
        wrong = x["gst_paise"] + rng.choice([-1, 1]) * rng.randint(x["gst_paise"] // 20 + 200, x["gst_paise"] // 5 + 500)
        diff = abs(wrong - x["gst_paise"])
        x["gst_paise"], x["total_paise"] = wrong, x["taxable_paise"] + wrong
        labels.append(("gst_mismatch", x["invoice_id"], diff))

    def mk_outlier(x):
        x["taxable_paise"] *= 5
        x["gst_paise"] = _gst(x["taxable_paise"], x["gst_rate_pct"])
        x["total_paise"] = x["taxable_paise"] + x["gst_paise"]
        labels.append(("amount_outlier", x["invoice_id"], x["total_paise"]))

    def mk_unbalanced(x):
        x["voucher_gap"] = rng.randint(100, 5000) * RUPEE
        labels.append(("unbalanced_voucher", x["voucher_id"], x["voucher_gap"]))

    def mk_duplicate(x):
        nonlocal n
        n += 1
        no = x["invoice_no"]
        variant = rng.choice([no.lower(), no.replace("/", "-"), " " + no + " "])
        dup = dict(x, invoice_id=f"INV-{n:04d}", voucher_id=f"VCH-{n:04d}", invoice_no=variant,
                   invoice_date=x["invoice_date"] + timedelta(days=rng.randint(1, 2)),
                   pay_split=None, voucher_gap=0, is_dup=True)
        used.add(dup["invoice_id"])
        dups.append(dup)
        labels.append(("duplicate_invoice", dup["invoice_id"], dup["total_paise"]))

    makers = dict(gst=mk_gst, outlier=mk_outlier, unbalanced=mk_unbalanced, duplicate=mk_duplicate)
    remaining = {t: counts[t] for t in makers}
    # Compound-risk vendors: three vendors get several anomaly TYPES each (ground truth for the
    # vendor-risk task). The rest are scattered over other vendors, one type per invoice.
    problem = rng.sample(sorted({x["vendor_id"] for x in ordinary}), 3)
    for vid, types in zip(problem, COMPOUND_PLAN):
        pool = [x for x in ordinary if x["vendor_id"] == vid]
        for t in types:
            if remaining[t] > 0:
                makers[t](pick(pool))
                remaining[t] -= 1
    rest = [x for x in ordinary if x["vendor_id"] not in problem]
    for t, k in remaining.items():
        for _ in range(k):
            makers[t](pick(rest))
    invoices += dups

    for _ in range(counts["split"]):
        if split_on_ordinary_vendors:
            x = pick(ordinary)
            target = int(limit * rng.uniform(1.1, 1.5))
            x["taxable_paise"] = target * 100 // (100 + x["gst_rate_pct"])
            x["gst_paise"] = _gst(x["taxable_paise"], x["gst_rate_pct"])
            x["total_paise"] = x["taxable_paise"] + x["gst_paise"]
        else:
            x = pick(largeinv)
        t = x["total_paise"]
        p1 = t // 2 + rng.randint(-t // 20, t // 20)
        parts = [p1, t - p1]
        assert all(config.SPLIT_MIN_FRACTION_PCT * limit // 100 <= p < limit for p in parts), parts
        x["pay_split"] = parts

    # Emit
    con = sqlite3.connect(ledger)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO vendors VALUES (?,?,?)", [(v["vendor_id"], v["name"], v["gstin"]) for v in vendors])
    pn = 0
    for x in invoices:
        con.execute("INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?)", (
            x["invoice_id"], x["vendor_id"], x["invoice_no"], x["invoice_date"].isoformat(),
            x["taxable_paise"], x["gst_rate_pct"], x["gst_paise"], x["total_paise"], x["voucher_id"]))
        credit = x["total_paise"] + x["voucher_gap"]
        con.executemany("INSERT INTO voucher_lines VALUES (?,?,?,?,?)", [
            (x["voucher_id"], 1, "Purchases", x["taxable_paise"], 0),
            (x["voucher_id"], 2, "Input GST", x["gst_paise"], 0),
            (x["voucher_id"], 3, "Accounts Payable", 0, credit)])
        if x.get("is_dup"):
            continue  # duplicates are never paid
        parts = x["pay_split"] or [x["total_paise"]]
        for j, amt in enumerate(parts):
            pn += 1
            pid = f"PAY-{pn:04d}"
            pd = x["invoice_date"] + timedelta(days=5 + j)
            con.execute("INSERT INTO payments VALUES (?,?,?,?,?)", (pid, x["vendor_id"], x["invoice_id"], pd.isoformat(), amt))
            if x["pay_split"]:
                labels.append(("payment_split", pid, amt))
    con.commit()
    con.close()

    labels_path.write_text(json.dumps({
        "seed": seed,
        "labels": [dict(detector=d, evidence_id=e, amount_paise=a) for d, e, a in labels]}, indent=1))
    return ledger, labels_path
