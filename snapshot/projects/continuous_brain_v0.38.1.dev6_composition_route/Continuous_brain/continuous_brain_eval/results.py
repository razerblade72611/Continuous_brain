"""Result records. The status vocabulary is enforced here, not by convention.

Every record carries the provenance the Joint Contract §6 requires. A record
may only say PASS or FAIL when (a) it names an APPROVED/FROZEN
preregistration, (b) it cites at least one primary artifact, and (c) it was
produced by a scientific runtime/substrate configuration. Everything else is
ENGINEERING_ONLY, UNREGISTERED_METRIC, CAPABILITY_ABSENT, or BLOCKED.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from continuous_brain_protocols import ResultStatus, canonical_json, scientific_claim_allowed, sha256_json
from continuous_brain_unified.prereg import PreregistrationError, PreregistrationRegistry

SPEC_FILE = "spec.json"


class StatusViolation(ValueError):
    pass


@dataclass
class ResultRecord:
    item: str                                   # e.g. "B4", "G2"
    status: ResultStatus
    metric_status: ResultStatus = ResultStatus.UNREGISTERED_METRIC
    substrate: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    prereg_id: str | None = None
    decision_rule: str | None = None
    notes: list[str] = field(default_factory=list)
    scientific_context: bool = False            # runtime.scientific / substrate built without fixtures

    def validate(self, registry: PreregistrationRegistry | None = None) -> None:
        """PASS/FAIL needs: a primary artifact (the run spec alone is not one), a preregistration that the
        registry itself says is APPROVED/FROZEN *for this item*, and a scientific context."""
        if self.status not in (ResultStatus.PASS, ResultStatus.FAIL):
            return
        artifacts = {k: v for k, v in (self.provenance.get("artifacts") or {}).items() if k != SPEC_FILE}
        if not scientific_claim_allowed(self.status, self.prereg_id, artifacts):
            raise StatusViolation(f"{self.item}: {self.status.value} needs a preregistration id and a primary artifact")
        try:
            (registry or PreregistrationRegistry.load()).require(self.prereg_id, item=self.item)
        except PreregistrationError as e:
            raise StatusViolation(f"{self.item}: {e}") from None
        if not self.scientific_context:
            raise StatusViolation(f"{self.item}: produced in a non-scientific context (fixtures or overrides)")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["metric_status"] = self.metric_status.value
        return d

    def to_json(self, registry: PreregistrationRegistry | None = None) -> str:
        self.validate(registry)
        return canonical_json(self.as_dict())

    def digest(self) -> str:
        return sha256_json(self.as_dict())
