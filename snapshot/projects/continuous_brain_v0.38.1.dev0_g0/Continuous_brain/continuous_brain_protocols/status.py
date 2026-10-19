"""Evidence and result status vocabulary.

`EvidenceStatus` is the dossier's classification of historical results.
`ResultStatus` is what a v0.38 result object may carry. A PASS or FAIL is
only allowed when the metric, margin, seed discipline, and decision rule are
preregistered (ChatGPT's clarification of C5); everything else is
engineering validation.
"""

from __future__ import annotations

from enum import Enum


class EvidenceStatus(str, Enum):
    ADMITTED_CLOSED_PASS = "ADMITTED/CLOSED PASS"
    DEV_PASS = "DEV PASS"
    FORMAL_FAIL = "FORMAL FAIL"
    DIAGNOSTIC = "DIAGNOSTIC"
    INFRASTRUCTURE_PASS = "INFRASTRUCTURE PASS"


class ResultStatus(str, Enum):
    ENGINEERING_ONLY = "ENGINEERING_ONLY"          # harness validation; no scientific content
    UNREGISTERED_METRIC = "UNREGISTERED_METRIC"    # a metric was computed but has no approved prereg
    CAPABILITY_ABSENT = "CAPABILITY_ABSENT"        # the substrate lacks what the item measures
    BLOCKED = "BLOCKED"                            # a prerequisite does not exist yet (e.g. E2 body)
    UNVERIFIED = "UNVERIFIED"                      # a claim without a cited primary artifact
    INFRASTRUCTURE_PASS = "INFRASTRUCTURE_PASS"
    PASS = "PASS"
    FAIL = "FAIL"


SCIENTIFIC_STATUSES = frozenset({ResultStatus.PASS, ResultStatus.FAIL})


def scientific_claim_allowed(status: ResultStatus, prereg_id: str | None, artifacts: dict | None) -> bool:
    """A PASS/FAIL needs an approved preregistration id and at least one primary artifact."""
    if status not in SCIENTIFIC_STATUSES:
        return True          # non-scientific statuses are always allowed
    return bool(prereg_id) and bool(artifacts)
