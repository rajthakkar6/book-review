"""Budget, tool allowlist, output truncation and the groundedness check."""
import json
import time
from collections.abc import Callable

from . import config
from .schemas import Finding, VendorRisk

ALLOWED_TOOLS = frozenset({"list_detectors", "run_detector", "get_invoice", "get_voucher", "vendor_summary"})


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_steps=config.MAX_STEPS, max_tool_calls=config.MAX_TOOL_CALLS,
                 max_tokens=config.MAX_TOKENS, max_wall_seconds=config.MAX_WALL_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self.max_steps, self.max_tool_calls = max_steps, max_tool_calls
        self.max_tokens, self.max_wall_seconds = max_tokens, max_wall_seconds
        self.clock = clock
        self.steps = self.tool_calls = self.tokens = 0
        self.started = clock()

    def start(self):
        self.started = self.clock()

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    def _wall(self):
        if self.elapsed > self.max_wall_seconds:
            raise BudgetExceeded(f"wall clock exceeded {self.max_wall_seconds}s")

    def step(self):
        self._wall()
        if self.steps + 1 > self.max_steps:
            raise BudgetExceeded(f"steps exceeded {self.max_steps}")
        self.steps += 1

    def tool_call(self, n: int = 1):
        self._wall()
        if self.tool_calls + n > self.max_tool_calls:
            raise BudgetExceeded(f"tool calls exceeded {self.max_tool_calls}")
        self.tool_calls += n

    def add_tokens(self, n: int):
        self.tokens += n
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(f"tokens exceeded {self.max_tokens}")


def truncate_output(text: str, max_chars: int = config.MAX_TOOL_OUTPUT_CHARS) -> str:
    """Shrink tool output to max_chars. JSON stays valid: trailing list items are dropped and counted."""
    if len(text) <= max_chars:
        return text
    try:
        obj = json.loads(text)
    except ValueError:
        return text[:max_chars]
    if not isinstance(obj, dict):
        return text[:max_chars]
    dropped = 0
    while True:
        obj["truncated_items"] = dropped
        out = json.dumps(obj, separators=(",", ":"))
        if len(out) <= max_chars:
            return out
        lists = [v for v in obj.values() if isinstance(v, list) and v]
        if not lists:
            return text[:max_chars]
        max(lists, key=len).pop()
        dropped += 1


class SeenEvidence:
    """Ids and amounts that tool outputs actually contained in this run.

    Only values under *_id / *_paise keys count, so ids mentioned in free text
    (e.g. a vendor name carrying an injection) never become 'seen'.
    """

    def __init__(self):
        self.ids: set[str] = set()
        self.ints: set[int] = set()
        self.paired: dict[str, int] = {}  # evidence_id -> amount_paise from detector records
        self.records: dict[str, dict[str, dict]] = {}  # detector -> evidence_id -> full record
        self.detectors_run: dict[str, tuple[int, int]] = {}  # detector -> (total_found, returned)

    def add_tool_output(self, text: str):
        try:
            self._walk(json.loads(text))
        except ValueError:
            pass

    def _walk(self, o):
        if isinstance(o, list):
            for x in o:
                self._walk(x)
        elif isinstance(o, dict):
            if isinstance(o.get("detector"), str) and isinstance(o.get("total_found"), int):
                self.detectors_run[o["detector"]] = (o["total_found"], o.get("returned", o["total_found"]))
            if isinstance(o.get("evidence_id"), str) and isinstance(o.get("amount_paise"), int):
                self.paired[o["evidence_id"]] = o["amount_paise"]
                if isinstance(o.get("detector"), str):
                    self.records.setdefault(o["detector"], {})[o["evidence_id"]] = o
            for k, v in o.items():
                if k.endswith("_id") or k == "duplicate_of":
                    if isinstance(v, str):
                        self.ids.add(v)
                elif k.endswith("_paise") and isinstance(v, int) and not isinstance(v, bool):
                    self.ints.add(v)
                else:
                    self._walk(v)


def check_grounded(findings: list[Finding], seen: SeenEvidence) -> list[str]:
    """Return error strings; empty means every cited id and amount came from a tool in this run."""
    errors = []
    for n, f in enumerate(findings, 1):
        missing = [i for i in f.evidence_ids if i not in seen.ids]
        if missing:
            errors.append(f"finding {n}: evidence ids not returned by any tool: {missing}")
            continue
        if all(i in seen.paired for i in f.evidence_ids):
            expected = sum(seen.paired[i] for i in f.evidence_ids)
            if f.amount_paise != expected:
                errors.append(f"finding {n}: amount {f.amount_paise} does not match tool amount {expected}")
        elif f.amount_paise not in seen.ints:
            errors.append(f"finding {n}: amount {f.amount_paise} was not returned by any tool")
    return errors


def check_vendor_risks(risks: list[VendorRisk], findings: list[Finding], seen: SeenEvidence) -> list[str]:
    """Each vendor risk must be grounded: vendor and evidence ids seen, evidence belongs to that vendor,
    types match the evidence, evidence is cited by a finding, risk level follows the 3+/2 rule."""
    cited = {i for f in findings for i in f.evidence_ids}
    detectors_of: dict[str, dict[str, str]] = {}  # evidence_id -> {detector: vendor_id}
    for det, recs in seen.records.items():
        for eid, rec in recs.items():
            detectors_of.setdefault(eid, {})[det] = rec.get("detail", {}).get("vendor_id")
    errors = []
    # completeness and order follow from the model's own findings
    types_by_vendor: dict[str, set[str]] = {}
    for f in findings:
        for eid in f.evidence_ids:
            vendor = seen.records.get(f.type, {}).get(eid, {}).get("detail", {}).get("vendor_id")
            if vendor:
                types_by_vendor.setdefault(vendor, set()).add(f.type)
    expected = {v for v, t in types_by_vendor.items() if len(t) >= 2}
    listed = [r.vendor_id for r in risks]
    if expected - set(listed):
        errors.append(f"vendor_risks is missing vendors whose findings span 2+ types: {sorted(expected - set(listed))}")
    if set(listed) - expected:
        errors.append(f"vendor_risks lists vendors that do not have findings of 2+ types: {sorted(set(listed) - expected)}")
    if len(set(listed)) != len(listed):
        errors.append("vendor_risks lists a vendor more than once")
    want = sorted(set(listed) & expected, key=lambda v: (-len(types_by_vendor[v]), v))
    if [v for v in listed if v in expected] != want:
        errors.append(f"vendor_risks order is wrong; required order (most distinct types first, ties by vendor_id): {want}")
    for n, r in enumerate(risks, 1):
        if r.vendor_id not in seen.ids:
            errors.append(f"vendor_risk {n}: vendor {r.vendor_id} was not returned by any tool")
            continue
        types = set()
        for eid in r.evidence_ids:
            owners = detectors_of.get(eid)
            if eid not in cited:
                errors.append(f"vendor_risk {n}: evidence {eid} is not cited by any finding")
            elif not owners or r.vendor_id not in owners.values():
                errors.append(f"vendor_risk {n}: evidence {eid} does not belong to vendor {r.vendor_id}")
            else:
                types |= {d for d, v in owners.items() if v == r.vendor_id}
        if not set(r.finding_types) <= types:
            errors.append(f"vendor_risk {n}: finding_types {sorted(set(r.finding_types) - types)} "
                          f"are not backed by the cited evidence")
        k = len(set(r.finding_types))
        if (k >= 3) != (r.risk == "high"):
            errors.append(f"vendor_risk {n}: risk must be 'high' for 3+ distinct types, 'medium' for 2 (got {k} types, {r.risk})")
    return errors


def check_clusters(findings: list[Finding], seen: SeenEvidence) -> list[str]:
    """A payment_split finding must cite exactly the members of ONE detector cluster (no merging, no partial)."""
    members: dict[str, set[str]] = {}
    for eid, rec in seen.records.get("payment_split", {}).items():
        members.setdefault(rec.get("detail", {}).get("cluster_id", eid), set()).add(eid)
    errors = []
    for n, f in enumerate(findings, 1):
        if f.type == "payment_split" and set(f.evidence_ids) not in members.values() and members:
            errors.append(f"finding {n}: payment_split must cite exactly the payments of ONE cluster_id "
                          f"(one finding per cluster); {sorted(f.evidence_ids)} is not a single cluster")
    return errors
