"""Sleep: the only phase in which anything learns.

One cycle:
  1. Select the most salient unconsolidated episodes.
  2. Snapshot the full state (verified) BEFORE touching anything.
  3. Baseline every regression probe.
  4. Bridge pass (backprop through the frozen backbone): train the latent pool
     and prefix so that, given an episode's stored features and prompt, the
     backbone better predicts the episode's target. Salience-weighted.
  5. Write-back pass (policy gradient): REINFORCE on logged currents, advantage
     = outcome - mean outcome.
  6. Core pass (local plasticity only): replay each episode's recorded input
     currents through the core with eligibility tracking, then apply the
     neuromodulator m = scale * (outcome - mean outcome).
  7. Re-run every probe. A component whose guarded probe regressed beyond
     tolerance ("bridge" = latent pool + prefix + write-back, or "core") is
     restored exactly from the pre-sleep snapshot; the other component keeps
     its update, so a core that fails its gate does not block the bridge from
     learning. After a partial rollback the probes are re-run; if anything
     still fails, everything is restored.
  8. If anything was kept, snapshot the result. Episodes are marked
     consolidated only when every component accepted its update; otherwise
     they stay eligible for the next sleep.
Every cycle appends a report to logs/sleep_log.jsonl.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field

import torch

from ..config import SleepConfig
from ..memory.episodic import Episode, EpisodicStore
from ..state.snapshot import SnapshotStore
from ..system import BrainComponents
from .regression import RegressionSuite

COMPONENTS = ("bridge", "core")


@dataclass
class SleepReport:
    sleep_id: str
    started_utc: float
    n_episodes: int
    skipped: bool = False
    baseline: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    rolled_back: bool = False
    rolled_back_components: list[str] = field(default_factory=list)
    accepted_components: list[str] = field(default_factory=list)
    after_partial_rollback: dict[str, float] = field(default_factory=dict)
    pre_snapshot: str | None = None
    post_snapshot: str | None = None
    stats: dict = field(default_factory=dict)


class SleepConsolidator:
    def __init__(self, system: BrainComponents, store: EpisodicStore, snapshots: SnapshotStore,
                 suite: RegressionSuite, config: SleepConfig | None = None):
        self.system = system
        self.store = store
        self.snapshots = snapshots
        self.suite = suite
        self.cfg = config or SleepConfig()
        self.log_path = snapshots.logs / "sleep_log.jsonl"

    # ------------------------------------------------------------------ passes

    def _bridge_pass(self, episodes: list[Episode], gen: torch.Generator) -> dict:
        eps = [e for e in episodes if e.target]
        if not eps:
            return {"skipped": "no episodes with a target"}
        bb, sys_ = self.system.backbone, self.system
        feats = torch.stack([torch.from_numpy(e.arrays["features"]) for e in eps])
        prompts = [bb.encode(e.prompt) for e in eps]
        targets = [bb.encode(e.target) for e in eps]
        weights = torch.tensor([max(e.salience, 1e-3) for e in eps])
        weights = weights / weights.sum()

        def full_loss() -> float:
            with torch.no_grad():
                nll = bb.lm_loss(sys_.prefix_from_features(feats), prompts, targets)
            return float((nll.cpu() * weights).sum())

        before = full_loss()
        opt = torch.optim.Adam(sys_.bridge_parameters(), lr=self.cfg.bridge_lr)
        sys_.latent_pool.train()
        sys_.prefix.train()
        with bb.gradient_mode():
            for _ in range(self.cfg.bridge_steps):
                idx = torch.multinomial(weights, min(self.cfg.bridge_batch, len(eps)),
                                        replacement=True, generator=gen)
                nll = bb.lm_loss(sys_.prefix_from_features(feats[idx]),
                                 [prompts[i] for i in idx], [targets[i] for i in idx])
                opt.zero_grad(set_to_none=True)
                nll.mean().backward()
                torch.nn.utils.clip_grad_norm_(sys_.bridge_parameters(), 1.0)
                opt.step()
        sys_.latent_pool.eval()
        sys_.prefix.eval()
        return {"n": len(eps), "loss_before": before, "loss_after": full_loss()}

    def _writeback_pass(self, episodes: list[Episode]) -> dict:
        eps = [e for e in episodes if e.outcome is not None]
        if len(eps) < 2:
            return {"skipped": "fewer than two episodes with an outcome"}
        outcomes = torch.tensor([float(e.outcome) for e in eps])
        adv = outcomes - outcomes.mean()
        if float(adv.abs().max()) < 1e-8:
            return {"skipped": "no outcome variance to learn from"}
        wb = self.system.writeback
        dev = next(wb.parameters()).device
        summaries = torch.stack([torch.from_numpy(e.arrays["wb_summary"]) for e in eps]).to(dev)
        currents = torch.stack([torch.from_numpy(e.arrays["wb_current"]) for e in eps]).to(dev)
        adv = (adv / (adv.std() + 1e-8)).to(dev)
        opt = torch.optim.Adam(wb.parameters(), lr=self.cfg.writeback_lr)
        wb.train()
        for _ in range(self.cfg.writeback_steps):
            loss = -(adv * wb.log_prob(summaries, currents)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        wb.eval()
        return {"n": len(eps), "final_loss": loss.item()}

    def _core_pass(self, episodes: list[Episode]) -> dict:
        eps = [e for e in episodes if e.outcome is not None and e.arrays["core_inputs"].size]
        if not eps:
            return {"skipped": "no episodes with an outcome and recorded core input"}
        core = self.system.core
        baseline = sum(float(e.outcome) for e in eps) / len(eps)
        applied = []
        for i, e in enumerate(eps):
            modulator = self.cfg.core_modulator_scale * (float(e.outcome) - baseline)
            if abs(modulator) < 1e-8:
                continue
            core.reset_state()
            core.reseed(self.cfg.seed + i)
            core.set_plasticity_tracking(True)
            try:
                for row in torch.from_numpy(e.arrays["core_inputs"]):
                    core.step(row[None].expand(core.batch_size, -1).clone(), e.cycle_s)
            finally:
                core.set_plasticity_tracking(False)
            applied.append({"episode": e.id, "modulator": modulator, **core.apply_plasticity(modulator)})
        core.reset_state()
        return {"n_replayed": len(applied), "replays": applied}

    # ------------------------------------------------------------------ cycle

    def run(self) -> SleepReport:
        report = SleepReport(sleep_id=f"sleep-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-"
                                      f"{uuid.uuid4().hex[:6]}",
                             started_utc=time.time(), n_episodes=0)
        episodes = self.store.select_for_sleep(self.cfg.max_episodes)
        report.n_episodes = len(episodes)
        if not episodes:
            report.skipped = True
            self._log(report)
            return report

        pre = self.snapshots.save_state({"state": self.system.full_state()}, label="pre-sleep",
                                        meta={"sleep_id": report.sleep_id})
        if not self.snapshots.verify(pre):
            raise RuntimeError(f"pre-sleep snapshot failed verification; refusing to sleep: {pre}")
        report.pre_snapshot = str(pre)
        pre_state = self.snapshots.load_state(pre, "state")

        gen = torch.Generator().manual_seed(self.cfg.seed)
        report.baseline = self.suite.evaluate(self.system, episodes)
        tol = self.cfg.regression_tolerance
        try:
            report.stats["bridge"] = self._bridge_pass(episodes, gen)
            report.stats["writeback"] = self._writeback_pass(episodes)
            report.stats["core"] = self._core_pass(episodes)
            report.after = self.suite.evaluate(self.system, episodes)
            result = self.suite.gate(report.baseline, report.after, tol)
            report.failures = list(result.failures)
            failed = set(result.failed_components)
            if failed:
                self._restore(pre_state, failed)
                report.after_partial_rollback = self.suite.evaluate(self.system, episodes)
                recheck = self.suite.gate(report.baseline, report.after_partial_rollback, tol)
                if not recheck.passed:
                    report.failures += [f"after partial rollback: {f}" for f in recheck.failures]
                    failed = set(COMPONENTS)
        except Exception as exc:
            report.failures.append(f"exception during sleep: {exc!r}")
            failed = set(COMPONENTS)

        if failed == set(COMPONENTS):
            self.system.load_full_state(pre_state)
        report.rolled_back_components = sorted(failed)
        report.accepted_components = sorted(set(COMPONENTS) - failed)
        report.rolled_back = bool(failed)
        report.passed = not failed

        if report.accepted_components:
            post = self.snapshots.save_state({"state": self.system.full_state()}, label="post-sleep",
                                             meta={"sleep_id": report.sleep_id,
                                                   "accepted": report.accepted_components})
            report.post_snapshot = str(post)
        if report.passed:
            self.store.mark_consolidated([e.id for e in episodes], report.sleep_id)
        self._log(report)
        return report

    def _restore(self, state: dict, components: set[str]) -> None:
        if "core" in components:
            self.system.core.load_state_dict(state["core"])
        if "bridge" in components:
            self.system.latent_pool.load_state_dict(state["latent_pool"])
            self.system.prefix.load_state_dict(state["prefix"])
            self.system.writeback.load_state_dict(state["writeback"])

    def _log(self, report: SleepReport) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(report), default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
