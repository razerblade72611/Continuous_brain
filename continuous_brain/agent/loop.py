"""The waking loop: the core runs continuously; the backbone is consulted.

Each `tick()` is one cognitive cycle:
  1. The core integrates for `cycle_s` with sensory input plus any pending
     write-back current (which decays after injection).
  2. Its projection spikes pass through the transducer: multiscale traces ->
     per-neuron features -> population features (+ synchrony).
  3. The invocation gate decides whether to consult the backbone. An incoming
     message always forces a consultation.
  4. A consultation: recall related past episodes, build the prompt, prepend
     the core's soft prefix, generate, decode the backbone's state into
     current for the core, and log the episode.

Nothing learns while awake. `sleep()` runs one consolidation cycle.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..bridge.backbone import Backbone
from ..bridge.gate import InvocationGate
from ..bridge.prefix import SoftPrefixBridge
from ..bridge.writeback import WritebackDecoder
from ..config import BrainConfig
from ..core.interface import SpikingCore
from ..memory.episodic import Episode, EpisodicStore
from ..sleep.consolidation import SleepConsolidator, SleepReport
from ..sleep.regression import (CoreHealthProbe, GeneralLanguageProbe, MatchedPairProbe, RecallProbe,
                                RegressionSuite)
from ..state.snapshot import SnapshotStore
from ..system import BrainComponents
from ..transducer import (FeatureAssembler, LatentTokenPool, MultiscaleTraceBank, PopulationAssignment,
                          PopulationPooling)

# General-language pairs for the regression gate: generic continuations the
# bridge's prefix must not make the backbone worse at.
DEFAULT_LANGUAGE_PAIRS: list[tuple[str, str]] = [
    ("The capital of France is", " Paris."),
    ("Water freezes at", " zero degrees Celsius."),
    ("One, two, three,", " four, five."),
    ("The opposite of hot is", " cold."),
    ("A week has", " seven days."),
    ("The sun rises in the", " east."),
]


@dataclass
class TickResult:
    t_s: float
    n_spikes: int
    gate_activation: float
    consulted: bool = False
    episode_id: str | None = None
    response: str | None = None


class ContinuousBrain:
    def __init__(self, core: SpikingCore, backbone: Backbone, root: str | os.PathLike,
                 config: BrainConfig | None = None, assignment: PopulationAssignment | None = None,
                 suite: RegressionSuite | None = None, seed: int = 0):
        self.cfg = cfg = config or BrainConfig()
        if core.batch_size != 1:
            raise ValueError("the waking loop runs a single individual (core batch_size must be 1)")
        if not math.isclose(cfg.transducer.bin_s, core.dt, rel_tol=1e-9):
            raise ValueError(f"transducer bin ({cfg.transducer.bin_s}) must equal the core's "
                             f"integration step ({core.dt})")
        self.root = Path(root)
        self.device = torch.device(cfg.loop.device)
        self.core, self.backbone = core, backbone
        self.snapshots = SnapshotStore(self.root)
        self.store = EpisodicStore(self.root, cfg.memory)

        tc = cfg.transducer
        if assignment is None:
            regions = core.projection_regions()
            assignment = (PopulationAssignment.from_regions(regions) if regions
                          else PopulationAssignment.balanced(core.n_projection, tc.n_populations, seed))
        self.assignment = assignment
        self.traces = MultiscaleTraceBank(core.n_projection, 1, tc.taus_s, tc.bin_s, tc.recency_tau_s,
                                          self.device)
        self.traces.reset(core.time_s)
        self.features = FeatureAssembler(self.traces.n_timescales)
        self.pooling = PopulationPooling(assignment)
        n_channels = self.pooling.n_out_channels(self.features.n_channels)

        torch.manual_seed(seed)
        latent_pool = LatentTokenPool(n_channels, assignment.n_populations, tc.n_latent_tokens,
                                      tc.latent_dim, tc.n_heads, tc.dropout).to(self.device).eval()
        prefix = SoftPrefixBridge(tc.latent_dim, backbone.hidden_size, cfg.bridge.prefix_mlp_hidden,
                                  backbone.embedding_rms()).to(self.device).eval()
        writeback = WritebackDecoder(backbone.hidden_size, core.n_input, cfg.bridge.writeback_hidden,
                                     cfg.bridge.max_current, cfg.bridge.writeback_noise_std
                                     ).to(self.device).eval()
        self.system = BrainComponents(core, backbone, latent_pool, prefix, writeback)
        self.gate = InvocationGate(cfg.gate.trigger_populations, cfg.gate.threshold, cfg.gate.refractory_s)
        self.suite = suite or RegressionSuite([GeneralLanguageProbe(DEFAULT_LANGUAGE_PAIRS), RecallProbe(),
                                               CoreHealthProbe(), MatchedPairProbe()])
        self.sleeper = SleepConsolidator(self.system, self.store, self.snapshots, self.suite, cfg.sleep)

        self.pending_current = torch.zeros(1, core.n_input, device=self.device)
        self.input_history: deque[torch.Tensor] = deque(maxlen=cfg.loop.history_cycles)
        self.wb_gen = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.last_pooled: torch.Tensor | None = None

    # ------------------------------------------------------------------ waking

    def tick(self, sensory: torch.Tensor | None = None, message: str | None = None) -> TickResult:
        cyc = self.cfg.loop.cycle_s
        inp = self.pending_current.clone()
        if sensory is not None:
            inp = inp + sensory.to(self.device).reshape(1, -1)
        events = self.core.step(inp, cyc)
        self.input_history.append(inp[0].detach().cpu())
        window = self.traces.advance(events, self.core.time_s)
        pooled = self.pooling(self.features(window), window)
        self.last_pooled = pooled
        self.pending_current = self.pending_current * math.exp(-cyc / self.cfg.bridge.writeback_decay_s)

        result = TickResult(self.core.time_s, events.n_events, float(self.gate.activation(pooled)[0]))
        if self.gate(pooled, self.core.time_s, force=message is not None):
            result.consulted = True
            result.episode_id, result.response = self._consult(pooled, message)
        return result

    def run(self, n_cycles: int, sensory: torch.Tensor | None = None) -> list[TickResult]:
        return [self.tick(sensory) for _ in range(n_cycles)]

    def _format_prompt(self, query: str, recalled: list[tuple[Episode, float]]) -> str:
        if not recalled:
            return query
        lines = ["Relevant memories:"]
        for ep, _ in recalled:
            answer = ep.target or ep.response
            lines.append(f"- {ep.query} -> {answer}")
        lines.append("")
        lines.append(query)
        return "\n".join(lines)

    @torch.no_grad()
    def _consult(self, pooled: torch.Tensor, message: str | None) -> tuple[str, str]:
        bb, mc = self.backbone, self.cfg.memory
        query = message if message is not None else self.cfg.loop.internal_prompt
        key = bb.summarize(query).cpu().numpy()
        recalled = self.store.search(key, mc.recall_k, mc.recall_min_similarity)
        prompt = self._format_prompt(query, recalled)
        prefix = self.system.prefix_from_features(pooled)[0]
        gen = bb.generate(prefix, prompt, self.cfg.loop.max_new_tokens)

        summary = WritebackDecoder.summarize(gen.hidden).to(self.device)
        current, _ = self.system.writeback.sample(summary[None], self.wb_gen)
        self.pending_current = self.pending_current + current

        # Replay record: the inputs that led here, then the write-back's decay.
        decay = math.exp(-self.cfg.loop.cycle_s / self.cfg.bridge.writeback_decay_s)
        pending = self.pending_current[0].detach().cpu()
        after = [pending * decay ** k for k in range(self.cfg.loop.aftermath_cycles)]
        core_inputs = torch.stack(list(self.input_history) + after).numpy()

        eid = self.store.add(query=query, prompt=prompt, response=gen.text, t_sim=self.core.time_s,
                             cycle_s=self.cfg.loop.cycle_s, features=pooled[0].cpu().numpy(), key=key,
                             wb_summary=summary.cpu().numpy(), wb_current=current[0].cpu().numpy(),
                             core_inputs=core_inputs)
        return eid, gen.text

    def feedback(self, episode_id: str, *, outcome: float | None = None, target: str | None = None,
                 important: bool | None = None) -> None:
        """Attach what happened after an episode: a scalar outcome and/or the text
        the response should have been. Appended, never edited in place."""
        self.store.annotate(episode_id, outcome=outcome, target=target, explicit=important)

    # ------------------------------------------------------------------ sleep & state

    def sleep(self) -> SleepReport:
        report = self.sleeper.run()
        # Waking starts from fresh transient state, as after biological sleep.
        self.core.reset_state()
        self.traces.reset(self.core.time_s)
        self.features.reset()
        self.pending_current.zero_()
        self.input_history.clear()
        return report

    def save_checkpoint(self, label: str = "manual") -> Path:
        return self.snapshots.save_state({"state": self.system.full_state(),
                                          "loop": {"gate": self.gate.state_dict()}}, label=label)

    def load_checkpoint(self, snapshot: str | os.PathLike) -> None:
        self.system.load_full_state(self.snapshots.load_state(snapshot, "state"))
        self.gate.load_state_dict(self.snapshots.load_state(snapshot, "loop")["gate"])
        self.traces.reset(self.core.time_s)
        self.features.reset()
        self.pending_current.zero_()
        self.input_history.clear()
