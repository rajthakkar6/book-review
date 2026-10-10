"""Deterministic findings built directly from detector records.

Used (a) as the eval baseline the agent must beat on value-add and (b) as the partial result when
the model runs out of budget. Grounded by construction: only fields the tools returned.
"""
from collections import defaultdict

from .schemas import Finding, VendorRisk


def _explain(r: dict) -> str:
    d, det = r.get("detail", {}), r["detector"]
    if det == "duplicate_invoice":
        return f"Invoice {r['evidence_id']} duplicates {d.get('duplicate_of')} (same vendor, number or date/total)."
    if det == "gst_mismatch":
        return (f"Recorded GST {d.get('recorded_gst_paise')} paise differs from expected "
                f"{d.get('expected_gst_paise')} paise.")
    if det == "unbalanced_voucher":
        return f"Voucher debits {d.get('debit_paise')} paise do not equal credits {d.get('credit_paise')} paise."
    if det == "amount_outlier":
        return f"Invoice total is far above vendor {d.get('vendor_id')}'s median of {d.get('vendor_median_paise')} paise."
    return f"{det} flagged {r['evidence_id']}."


def baseline_findings(records_by_detector: dict[str, dict[str, dict]] | dict[str, list[dict]]) -> list[Finding]:
    out = []
    for det, recs in records_by_detector.items():
        recs = list(recs.values()) if isinstance(recs, dict) else list(recs)
        if det == "payment_split":
            clusters = defaultdict(list)
            for r in recs:
                clusters[r["detail"].get("cluster_id", r["evidence_id"])].append(r)
            for cid, rs in sorted(clusters.items()):
                out.append(Finding(
                    type="payment_split", evidence_ids=sorted(r["evidence_id"] for r in rs),
                    amount_paise=sum(r["amount_paise"] for r in rs),
                    explanation=f"{len(rs)} payments to vendor {rs[0]['detail'].get('vendor_id')} within a short window, "
                                f"each under the approval limit, total {sum(r['amount_paise'] for r in rs)} paise."))
        else:
            out += [Finding(type=det, evidence_ids=[r["evidence_id"]], amount_paise=r["amount_paise"],
                            explanation=_explain(r)) for r in sorted(recs, key=lambda r: r["evidence_id"])]
    return out


def baseline_vendor_risks(records_by_detector) -> list[VendorRisk]:
    """Group records by vendor; vendors with 2+ distinct detector types, most types first, ties by id."""
    by_vendor = defaultdict(lambda: defaultdict(list))
    for det, recs in records_by_detector.items():
        recs = recs.values() if isinstance(recs, dict) else recs
        for r in recs:
            by_vendor[r["detail"].get("vendor_id")][det].append(r["evidence_id"])
    out = []
    for vid, per in by_vendor.items():
        if vid is None or len(per) < 2:
            continue
        types = sorted(per)
        ids = sorted(i for v in per.values() for i in v)
        out.append(VendorRisk(
            vendor_id=vid, risk="high" if len(types) >= 3 else "medium", finding_types=types, evidence_ids=ids,
            action=f"Review all {len(ids)} flagged items for {vid} together before approving further payments.",
            rationale=f"{vid} has anomalies of {len(types)} different types: {', '.join(types)}."))
    return sorted(out, key=lambda r: (-len(r.finding_types), r.vendor_id))


def vendor_risks_from_findings(findings: list[Finding], records: dict[str, dict[str, dict]]) -> list[VendorRisk]:
    """Code-built vendor-risk skeleton (vendors, level, types, evidence, order) from the validated findings.
    Text fields are template placeholders; the model may replace them with better wording."""
    cited: dict[str, dict[str, dict]] = {}
    for f in findings:
        for eid in f.evidence_ids:
            rec = records.get(f.type, {}).get(eid)
            if rec:
                cited.setdefault(f.type, {})[eid] = rec
    return baseline_vendor_risks(cited)


def finding_key(f: Finding, records: dict[str, dict[str, dict]]) -> tuple[str, str]:
    """(type, id) a model explanation is keyed by: evidence_id, or cluster_id for payment_split."""
    if f.type == "payment_split":
        rec = records.get("payment_split", {}).get(f.evidence_ids[0], {})
        return f.type, rec.get("detail", {}).get("cluster_id", f.evidence_ids[0])
    return f.type, f.evidence_ids[0]
