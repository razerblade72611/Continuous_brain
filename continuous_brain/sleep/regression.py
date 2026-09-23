"""The regression gate: the only thing standing between sleep and silent damage.

Probes are evaluated before and after each sleep cycle with identical inputs
and identical noise seeds, so the difference measures only what sleep changed.
A guarded probe may not drop by more than its tolerance, or the component it
guards ("bridge" or "core") is rolled back to its pre-sleep state.

Probes are side-effect free: each saves and restores the core's full state.

Built-in probes:
  GeneralLanguageProbe  guarded   Does the bridge's prefix damage the backbone's
                                  general language ability? (-NLL on fixed texts)
  RecallProbe           reported  How well does the system reproduce targets of
                                  the episodes being consolidated? (-NLL)
  CoreHealthProbe       guarded   Is the core's spontaneous firing still inside a
                                  healthy band (not silent, not runaway)?
  MatchedPairProbe      guarded   Does the core still discriminate mirror-image
                                  input pairs? d' across independent noise seeds.
                                  This is the probe aimed at the core's OWN
                                  weights (spec: InfantBrain's role).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..memory.episodic import Episode
from ..system import BrainComponents


class Probe(ABC):
    name: str = "probe"
    guarded: bool = True
    tolerance: float | None = None   # None -> suite default
    #: Which learned component this probe guards: "bridge" (latent pool, prefix,
    #: write-back) or "core". A failure rolls back only that component.
    component: str = "bridge"

    @abstractmethod
    def evaluate(self, system: BrainComponents, episodes: list[Episode]) -> float:
        """Higher is better."""


def _features(episodes: list[Episode]) -> list[torch.Tensor]:
    return [torch.from_numpy(e.arrays["features"]) for e in episodes if e.arrays is not None]


class GeneralLanguageProbe(Probe):
    name = "general_language"
    guarded = True

    def __init__(self, pairs: list[tuple[str, str]], reference_features: list[torch.Tensor] | None = None):
        if not pairs:
            raise ValueError("GeneralLanguageProbe needs at least one (prompt, target) pair")
        self.pairs = pairs
        self.reference_features = reference_features

    @torch.no_grad()
    def evaluate(self, system: BrainComponents, episodes: list[Episode]) -> float:
        feats = self.reference_features or _features(episodes)
        if not feats:
            return 0.0
        bb = system.backbone
        f = torch.stack([feats[i % len(feats)] for i in range(len(self.pairs))])
        prefix = system.prefix_from_features(f)
        nll = bb.lm_loss(prefix, [bb.encode(p) for p, _ in self.pairs],
                         [bb.encode(t) for _, t in self.pairs])
        return -float(nll.mean())


class RecallProbe(Probe):
    name = "episode_recall"
    guarded = False
    component = "bridge"

    @torch.no_grad()
    def evaluate(self, system: BrainComponents, episodes: list[Episode]) -> float:
        eps = [e for e in episodes if e.target and e.arrays is not None]
        if not eps:
            return 0.0
        bb = system.backbone
        prefix = system.prefix_from_features(torch.stack(_features(eps)))
        nll = bb.lm_loss(prefix, [bb.encode(e.prompt) for e in eps], [bb.encode(e.target) for e in eps])
        return -float(nll.mean())


class _CoreProbe(Probe):
    """Runs the core from clean transient state with fixed seeds, then restores it."""

    component = "core"

    def _run(self, core, current: torch.Tensor, duration_s: float, seed: int) -> torch.Tensor:
        core.reset_state()
        core.reseed(seed)
        ev = core.step(current, duration_s)
        counts = torch.zeros(core.batch_size, core.n_projection)
        if ev.n_events:
            counts.index_put_((ev.batch.cpu(), ev.neuron.cpu()), ev.value.cpu(), accumulate=True)
        return counts

    def evaluate(self, system: BrainComponents, episodes: list[Episode]) -> float:
        core = system.core
        saved = core.state_dict()
        tracking = getattr(core, "tracking", False)
        core.set_plasticity_tracking(False)
        try:
            return self._score(core)
        finally:
            core.load_state_dict(saved)
            core.set_plasticity_tracking(tracking)

    @abstractmethod
    def _score(self, core) -> float: ...


class CoreHealthProbe(_CoreProbe):
    name = "core_health"
    guarded = True
    tolerance = 0.0

    def __init__(self, low_hz: float = 0.1, high_hz: float = 50.0, duration_s: float = 0.5, seed: int = 1234):
        self.low_hz, self.high_hz = low_hz, high_hz
        self.duration_s, self.seed = duration_s, seed

    def _score(self, core) -> float:
        counts = self._run(core, torch.zeros(core.batch_size, core.n_input), self.duration_s, self.seed)
        rate = float(counts.mean()) / self.duration_s
        if self.low_hz <= rate <= self.high_hz:
            return 0.0
        edge = self.low_hz if rate < self.low_hz else self.high_hz
        return -abs(math.log(max(rate, 1e-6) / edge))


class MatchedPairProbe(_CoreProbe):
    """Mean d' between the core's responses to each member of mirror-image input pairs.

    Each condition is run over `n_trials` independent noise seeds; seeds are
    fixed across evaluations. Trials, not repeated presentations of one seed,
    are the unit of replication.
    """

    name = "matched_pair_discrimination"
    guarded = True

    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None, n_trials: int = 8,
                 duration_s: float = 0.1, amplitude: float = 1.5, seed: int = 99):
        self.pairs = pairs
        self.n_trials, self.duration_s = n_trials, duration_s
        self.amplitude, self.seed = amplitude, seed

    def default_pairs(self, n_input: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        half = n_input // 2
        left = torch.zeros(n_input)
        left[:half] = self.amplitude
        ramp = torch.linspace(0.0, self.amplitude, n_input)
        return [(left, left.flip(0)), (ramp, ramp.flip(0))]

    def _score(self, core) -> float:
        pairs = self.pairs or self.default_pairs(core.n_input)
        dprimes = []
        for p, (a, b) in enumerate(pairs):
            resp = {}
            for name, pattern in (("a", a), ("b", b)):
                rows = []
                for trial in range(self.n_trials):
                    cur = pattern[None].expand(core.batch_size, -1).clone()
                    rows.append(self._run(core, cur, self.duration_s, self.seed + 1000 * p + trial)[0])
                resp[name] = torch.stack(rows)                     # [trials, n_projection]
            diff = resp["a"].mean(0) - resp["b"].mean(0)
            pooled_var = 0.5 * (resp["a"].var(0) + resp["b"].var(0)).mean()
            dprimes.append(float(diff.norm() / torch.sqrt(pooled_var * diff.numel() + 1e-6)))
        return sum(dprimes) / len(dprimes)


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    failed_components: set[str]


class RegressionSuite:
    def __init__(self, probes: list[Probe]):
        names = [p.name for p in probes]
        if len(set(names)) != len(names):
            raise ValueError("probe names must be unique")
        self.probes = probes

    def evaluate(self, system: BrainComponents, episodes: list[Episode]) -> dict[str, float]:
        return {p.name: p.evaluate(system, episodes) for p in self.probes}

    def gate(self, baseline: dict[str, float], after: dict[str, float], tolerance: float) -> GateResult:
        failures, components = [], set()
        for p in self.probes:
            if not p.guarded:
                continue
            tol = tolerance if p.tolerance is None else p.tolerance
            if after[p.name] < baseline[p.name] - tol:
                failures.append(f"{p.name}: {baseline[p.name]:.4f} -> {after[p.name]:.4f} (tolerance {tol})")
                components.add(p.component)
        return GateResult(not failures, failures, components)
