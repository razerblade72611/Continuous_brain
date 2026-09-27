"""Versioned decision rules, kept separate from experiment configuration.

An experiment names its rule by id; the rule's implementation and the SHA-256
of its source are recorded with the result, so changing a rule can never
silently change an old result's meaning. Code support for a PROPOSED rule is
not approval of it (Joint Contract §9).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from continuous_brain_protocols import sha256_bytes

from .stats import bootstrap_mean_ci, wilcoxon_signed_rank_one_sided


@dataclass(frozen=True)
class DecisionRule:
    rule_id: str
    status: str                    # CURRENT | PROPOSED | DRAFT
    description: str
    source: str                    # Linear document / issue
    fn: Callable[..., Any]
    depends_on: tuple[str, ...] = ()   # modules whose code decides the outcome

    def source_sha256(self) -> str:
        """Hash of the wrapper *and* the full source files of every module the rule depends on."""
        import importlib
        parts = [inspect.getsource(self.fn).encode()]
        for name in self.depends_on:
            parts.append(name.encode() + b"\0" + inspect.getsource(importlib.import_module(name)).encode())
        return sha256_bytes(b"\0\0".join(parts))

    def record(self) -> dict:
        return {"rule_id": self.rule_id, "status": self.status, "source": self.source,
                "implementation_sha256": self.source_sha256()}


def _dav32_v1(a, b):
    from continuous_brain_snn.parity import compare
    return compare(a, b)


def _a1_proposed(a, b, **kw):
    from .parity_a1 import compare_a1
    return compare_a1(a, b, **kw)


def _a1b_proposed(a, b, **kw):
    from .parity_a1 import compare_a1b
    return compare_a1b(a, b, **kw)


def _e2_primary(delta_r2_per_replicate, threshold: float = 0.05, n_resamples: int = 10_000, seed: int = 0) -> dict:
    """E2 prereg: PASS iff mean dR2 >= threshold and the 95 % bootstrap CI lower bound > 0."""
    d = np.asarray(delta_r2_per_replicate, dtype=np.float64)
    ci = bootstrap_mean_ci(d, n_resamples, 0.95, seed)
    return {"mean_delta_r2": ci.estimate, "ci": ci.as_dict(), "wilcoxon_one_sided_p": wilcoxon_signed_rank_one_sided(d),
            "criterion_met": bool(ci.estimate >= threshold and ci.low > 0), "threshold": threshold}


RULES: dict[str, DecisionRule] = {
    "DAV-32-v1": DecisionRule("DAV-32-v1", "CURRENT",
                              "cross-backend parity: per-population rate +-5 %, ISI CV +-0.05, correlation KS < 0.1",
                              "Linear DAV-32", _dav32_v1, ("continuous_brain_snn.parity",)),
    "A1-proposed": DecisionRule("A1-proposed", "PROPOSED",
                                "power-aware parity: tolerance max(5 %, 3.57 SE), bootstrap CV SE, KS < 0.1",
                                "Linear doc 'Proposed Amendment A1' (A1.2 original draft)", _a1_proposed,
                                ("continuous_brain_eval.parity_a1", "continuous_brain_snn.parity")),
    "A1b-proposed": DecisionRule("A1b-proposed", "PROPOSED",
                                 "power-aware parity with 280-test Bonferroni family: z=3.74755 (~3.75)",
                                 "ChatGPT/Claude review 2026-09-27; requires David approval before scientific use",
                                 _a1b_proposed, ("continuous_brain_eval.parity_a1", "continuous_brain_snn.parity")),
    "E2-primary-v1": DecisionRule("E2-primary-v1", "DRAFT",
                                  "E2 primary endpoint: mean dR2(DEV - max baseline) >= 0.05 and CI lower bound > 0",
                                  "Linear doc 'E2 Preregistration & Decision Rule' (Draft v1)", _e2_primary,
                                  ("continuous_brain_eval.stats",)),
}


def rule(rule_id: str) -> DecisionRule:
    try:
        return RULES[rule_id]
    except KeyError:
        raise KeyError(f"unknown decision rule {rule_id!r}; known: {sorted(RULES)}") from None
