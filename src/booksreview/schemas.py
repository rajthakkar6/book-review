from typing import Literal

from pydantic import BaseModel, Field

FindingType = Literal["duplicate_invoice", "gst_mismatch", "unbalanced_voucher", "payment_split", "amount_outlier"]
Status = Literal["ok", "refused", "budget_exceeded", "insufficient_evidence", "llm_error"]


class Finding(BaseModel):
    type: FindingType
    evidence_ids: list[str] = Field(min_length=1)
    amount_paise: int  # integer paise, never float
    explanation: str = Field(min_length=1)


class VendorRisk(BaseModel):
    """Cross-detector correlation: a vendor with anomalies of 2+ different types.
    risk: high = 3+ distinct types, medium = exactly 2."""
    vendor_id: str
    risk: Literal["high", "medium"]
    finding_types: list[FindingType] = Field(min_length=2)
    evidence_ids: list[str] = Field(min_length=1)
    action: str = Field(min_length=1)  # one concrete next step for the reviewer
    rationale: str = Field(min_length=1)


class Report(BaseModel):
    status: Status
    findings: list[Finding] = []
    vendor_risks: list[VendorRisk] = []  # ordered: most distinct types first, ties by vendor_id
    summary: str = ""
    partial: bool = False  # findings were built by code from detector output, not written by the model
