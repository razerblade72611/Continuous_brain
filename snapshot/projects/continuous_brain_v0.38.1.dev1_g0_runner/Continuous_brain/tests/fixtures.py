"""Test-only components. They carry TEST_FIXTURE provenance, so a runtime only
accepts them under BindingPolicy.TEST_FIXTURES, which marks it non-scientific.
They exist to exercise the composition engine; they are not mechanisms."""

from __future__ import annotations

import numpy as np

from continuous_brain_protocols import ChannelKind, ChannelSpec, StreamSpec
from continuous_brain_unified import (BindingPolicy, ChannelDecl, Component, ComponentProvenance, ProvenanceKind,
                                      RuntimeConfig, UnifiedRuntime)

OBS = StreamSpec((ChannelSpec("joint_angle", (2,), "rad", ChannelKind.SENSOR),), 0.01)


class Accumulator(Component):
    """Writes (sum of its inputs) + its own tick count to each declared channel."""

    def __init__(self, name, slot, writes, reads=(), delayed=(), shape=(2,), exact=False, uses_coprocessor=False):
        self.name, self.slot = name, slot
        self.reads, self.reads_delayed = tuple(reads), tuple(delayed)
        self.writes = tuple(ChannelDecl(w, slot, shape, exact_contract=exact) for w in writes)
        self.uses_coprocessor = uses_coprocessor
        self.count = 0
        self.replies = 0

    def provenance(self):
        return ComponentProvenance(ProvenanceKind.TEST_FIXTURE, {"fixture": type(self).__name__})

    def step(self, ctx):
        self.count += 1
        total = sum(float(v.sum()) for v in (*ctx.inputs.values(), *ctx.delayed.values()))
        if ctx.coprocessor is not None:
            resp = ctx.coprocessor.query(self.name, "describe", {"tick": ctx.tick})
            self.replies += int(resp.available)
            total += 100.0 * resp.available
        return {d.name: np.full(d.shape, total + self.count) for d in self.writes}

    def state_dict(self):
        return {"count": self.count, "replies": self.replies}

    def load_state_dict(self, state):
        self.count, self.replies = state["count"], state["replies"]


def observation(seed, tick):
    rng = np.random.default_rng(seed * 100_003 + tick)
    return {"joint_angle": rng.uniform(-1, 1, 2)}


def fixture_runtime(seed=0, coprocessor=None, enabled=False, with_autobiography=True):
    rt = UnifiedRuntime(RuntimeConfig(seed=seed, binding_policy=BindingPolicy.TEST_FIXTURES, observation=OBS,
                                      coprocessor_enabled=enabled), coprocessor=coprocessor)
    rt.bind(Accumulator("wm", "working_memory", ["working_memory.contents"], reads=["environment.joint_angle"],
                        exact=True))
    if with_autobiography:
        rt.bind(Accumulator("auto", "autobiography", ["autobiography.record", "autobiography.discrepancy"],
                            reads=["working_memory.contents"], delayed=["autobiography.record"], exact=True))
        rt.bind(Accumulator("self", "self_state", ["self_state.vector"], reads=["autobiography.discrepancy"],
                            uses_coprocessor=True, exact=True))
    return rt
