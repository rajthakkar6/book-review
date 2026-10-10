"""The only module that touches the ledger. Read-only by construction."""
import sqlite3
from pathlib import Path

from . import config


def connect_ro(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the ledger read-only. Enforced twice: URI mode=ro and PRAGMA query_only."""
    p = Path(path or config.LEDGER_PATH).resolve()
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params)]


def get_invoice(conn, invoice_id: str) -> dict | None:
    r = rows(conn, "SELECT * FROM invoices WHERE invoice_id=?", (invoice_id,))
    return r[0] if r else None


def get_voucher(conn, voucher_id: str) -> dict | None:
    lines = rows(conn, "SELECT * FROM voucher_lines WHERE voucher_id=? ORDER BY line_no", (voucher_id,))
    if not lines:
        return None
    return {
        "voucher_id": voucher_id,
        "lines": lines,
        "debit_paise": sum(l["debit_paise"] for l in lines),
        "credit_paise": sum(l["credit_paise"] for l in lines),
    }


def vendor_summary(conn, vendor_id: str) -> dict | None:
    v = rows(conn, "SELECT * FROM vendors WHERE vendor_id=?", (vendor_id,))
    if not v:
        return None
    agg = rows(
        conn,
        "SELECT COUNT(*) n, COALESCE(SUM(total_paise),0) total FROM invoices WHERE vendor_id=?",
        (vendor_id,),
    )[0]
    pay = rows(
        conn,
        "SELECT COUNT(*) n, COALESCE(SUM(amount_paise),0) total FROM payments WHERE vendor_id=?",
        (vendor_id,),
    )[0]
    return {
        **v[0],
        "invoice_count": agg["n"],
        "invoiced_paise": agg["total"],
        "payment_count": pay["n"],
        "paid_paise": pay["total"],
    }
