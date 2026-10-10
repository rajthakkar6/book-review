"""Deterministic anomaly detectors. Each record: detector, evidence_id, amount_paise, detail."""
import re
import statistics
from collections import defaultdict
from datetime import date, timedelta

from . import config
from .db import rows


def _rec(detector, evidence_id, amount_paise, **detail):
    return dict(detector=detector, evidence_id=evidence_id, amount_paise=int(amount_paise), detail=detail)


def duplicate_invoice(conn):
    """Same vendor + normalised invoice number, or same vendor + date + total. Later copies flagged."""
    inv = rows(conn, "SELECT * FROM invoices ORDER BY invoice_date, invoice_id")
    groups = defaultdict(list)
    for r in inv:
        groups[(r["vendor_id"], re.sub(r"[^a-z0-9]", "", r["invoice_no"].lower()))].append(r)
        groups[(r["vendor_id"], r["invoice_date"], r["total_paise"])].append(r)
    flagged = {}
    for g in groups.values():
        for r in g[1:]:
            flagged.setdefault(r["invoice_id"], g[0]["invoice_id"])
    by_id = {r["invoice_id"]: r for r in inv}
    return [_rec("duplicate_invoice", i, by_id[i]["total_paise"], vendor_id=by_id[i]["vendor_id"],
                 duplicate_of=orig) for i, orig in sorted(flagged.items())]


def gst_mismatch(conn):
    out = []
    for r in rows(conn, "SELECT * FROM invoices ORDER BY invoice_id"):
        expected = (r["taxable_paise"] * r["gst_rate_pct"] + 50) // 100
        diff = abs(r["gst_paise"] - expected)
        if diff > config.GST_TOLERANCE_PAISE:
            out.append(_rec("gst_mismatch", r["invoice_id"], diff, vendor_id=r["vendor_id"], expected_gst_paise=expected,
                            recorded_gst_paise=r["gst_paise"]))
    return out


def unbalanced_voucher(conn):
    return [_rec("unbalanced_voucher", r["voucher_id"], abs(r["d"] - r["c"]), vendor_id=r["vendor_id"],
                 debit_paise=r["d"], credit_paise=r["c"])
            for r in rows(conn, "SELECT l.voucher_id, SUM(l.debit_paise) d, SUM(l.credit_paise) c, i.vendor_id "
                                "FROM voucher_lines l JOIN invoices i ON i.voucher_id = l.voucher_id "
                                "GROUP BY l.voucher_id HAVING d != c ORDER BY l.voucher_id")]


def payment_split(conn):
    """Payments to one vendor inside a short window, each under the limit but sizeable, summing to >= limit."""
    limit = config.APPROVAL_LIMIT_PAISE
    floor = limit * config.SPLIT_MIN_FRACTION_PCT // 100
    by_vendor = defaultdict(list)
    for p in rows(conn, "SELECT * FROM payments WHERE amount_paise < ? AND amount_paise >= ? "
                        "ORDER BY pay_date, payment_id", (limit, floor)):
        by_vendor[p["vendor_id"]].append(p)
    out = []
    for vendor, ps in by_vendor.items():
        i = 0
        while i < len(ps):
            start = date.fromisoformat(ps[i]["pay_date"])
            cluster = [p for p in ps[i:] if date.fromisoformat(p["pay_date"]) <= start + timedelta(days=config.SPLIT_WINDOW_DAYS)]
            if len(cluster) >= 2 and sum(p["amount_paise"] for p in cluster) >= limit:
                total = sum(p["amount_paise"] for p in cluster)
                cid = f"{vendor}:{cluster[0]['payment_id']}"
                out += [_rec("payment_split", p["payment_id"], p["amount_paise"], vendor_id=vendor,
                             cluster_id=cid, cluster_total_paise=total) for p in cluster]
                i += len(cluster)
            else:
                i += 1
    return sorted(out, key=lambda r: r["evidence_id"])


def amount_outlier(conn):
    """Invoice total far above the vendor's median, in scaled MADs. Integer arithmetic only."""
    by_vendor = defaultdict(list)
    for r in rows(conn, "SELECT invoice_id, vendor_id, total_paise FROM invoices ORDER BY invoice_id"):
        by_vendor[r["vendor_id"]].append(r)
    out = []
    for vendor, invs in by_vendor.items():
        if len(invs) < config.OUTLIER_MIN_HISTORY:
            continue
        totals = [r["total_paise"] for r in invs]
        med = int(statistics.median_low(totals))
        mad = max(int(statistics.median_low([abs(t - med) for t in totals])), med // 100)  # floor: 1% of median
        for r in invs:
            # (t - med) > k * 1.4826 * mad, scaled to stay in integers
            if (r["total_paise"] - med) * 10_000 > config.OUTLIER_MAD_K * 14_826 * mad:
                out.append(_rec("amount_outlier", r["invoice_id"], r["total_paise"], vendor_id=vendor,
                                vendor_median_paise=med))
    return sorted(out, key=lambda r: r["evidence_id"])


DETECTORS = {
    "duplicate_invoice": (duplicate_invoice, "Same vendor and invoice number (normalised) or same vendor/date/total."),
    "gst_mismatch": (gst_mismatch, "Recorded GST differs from taxable * rate."),
    "unbalanced_voucher": (unbalanced_voucher, "Voucher debits do not equal credits."),
    "payment_split": (payment_split, "Several sub-limit payments to a vendor in a short window that together reach the approval limit."),
    "amount_outlier": (amount_outlier, "Invoice total far above the vendor's usual amount."),
}


def run(conn, name: str, limit: int | None = None) -> list[dict]:
    if name not in DETECTORS:
        raise KeyError(f"unknown detector: {name}")
    recs = DETECTORS[name][0](conn)
    names = {v["vendor_id"]: v["name"] for v in rows(conn, "SELECT vendor_id, name FROM vendors")}
    for r in recs:  # vendor name comes from ledger text: untrusted data
        r["detail"]["vendor_name"] = names.get(r["detail"].get("vendor_id"))
    return recs[:limit] if limit is not None else recs
