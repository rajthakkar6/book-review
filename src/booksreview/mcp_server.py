"""MCP tool layer. Five narrow, read-only tools; no arbitrary SQL."""
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from . import config, db, detectors

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def build_server(ledger_path: Path | str | None = None) -> MCPServer:
    server = MCPServer("books-review")

    def conn():  # fresh read-only connection per call; cheap and thread-safe
        return db.connect_ro(ledger_path)

    @server.tool(annotations=READ_ONLY)
    def list_detectors() -> dict:
        """List the available anomaly detectors and what each looks for."""
        return {"detectors": [{"name": n, "description": d} for n, (_, d) in detectors.DETECTORS.items()]}

    @server.tool(annotations=READ_ONLY)
    def run_detector(name: str, limit: int = config.MAX_LIMIT) -> dict:
        """Run one detector. Returns records with evidence_id and amount_paise. limit is capped at 50."""
        if name not in detectors.DETECTORS:
            raise ValueError(f"unknown detector '{name}'; use list_detectors")
        limit = max(1, min(limit, config.MAX_LIMIT))
        with conn() as c:
            recs = detectors.run(c, name)
        return {"detector": name, "total_found": len(recs), "returned": min(len(recs), limit),
                "records": recs[:limit]}

    @server.tool(annotations=READ_ONLY)
    def get_invoice(invoice_id: str) -> dict:
        """Fetch one invoice by id (e.g. INV-0001)."""
        with conn() as c:
            r = db.get_invoice(c, invoice_id)
        if r is None:
            raise ValueError(f"invoice not found: {invoice_id}")
        return r

    @server.tool(annotations=READ_ONLY)
    def get_voucher(voucher_id: str) -> dict:
        """Fetch a voucher's lines and debit/credit totals by id (e.g. VCH-0001)."""
        with conn() as c:
            r = db.get_voucher(c, voucher_id)
        if r is None:
            raise ValueError(f"voucher not found: {voucher_id}")
        return r

    @server.tool(annotations=READ_ONLY)
    def vendor_summary(vendor_id: str) -> dict:
        """Vendor details with invoice and payment totals (e.g. V001)."""
        with conn() as c:
            r = db.vendor_summary(c, vendor_id)
        if r is None:
            raise ValueError(f"vendor not found: {vendor_id}")
        return r

    return server
