"""The fifteen required subsystems (Joint Contract §1).

Twelve are *mechanism* slots: each is either bound to a component whose
provenance the runtime accepts (a hash-verified legacy adapter, C1) or is
reported UNBOUND. The runtime never substitutes a stand-in.

Three are *infrastructure* slots the runtime provides itself (persistence,
provenance, the coprocessor interface). They carry no cognitive claim.

Each mechanism slot names canonical output channels. A bound component must
write at least these, so interventions and gate evaluators can target
"the autobiography" without knowing which implementation is bound. Channel
names are namespaced by slot: only the slot's component may write
`<slot>.<anything>`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SlotKind(str, Enum):
    MECHANISM = "mechanism"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class SlotSpec:
    name: str
    kind: SlotKind
    description: str
    canonical_writes: tuple[str, ...] = ()
    legacy_lineage: tuple[str, ...] = ()        # dossier lines that may supply a mechanism (DAV-74 decides)


REQUIRED_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("perception", SlotKind.MECHANISM, "observation / perception adapter",
             ("perception.features",), ("v0.9 dual visual streams", "v0.26 discovery embodiment admission")),
    SlotSpec("workspace", SlotKind.MECHANISM, "recurrent state / workspace interface",
             ("workspace.state",), ("persistent recurrent workspace",)),
    SlotSpec("working_memory", SlotKind.MECHANISM, "limited-capacity working memory",
             ("working_memory.contents",), ("v0.13-v0.19 admitted causal working-memory components",)),
    SlotSpec("episodic_memory", SlotKind.MECHANISM, "episodic memory",
             ("episodic_memory.slots",), ("v0.23 two-item ordered episodic memory",)),
    SlotSpec("self_state", SlotKind.MECHANISM, "persistent self-state",
             ("self_state.vector",), ("v0.28 persistent self + autobiography", "v0.31 recurrent self-trajectory")),
    SlotSpec("autobiography", SlotKind.MECHANISM, "autobiography / expected-vs-actual discrepancy",
             ("autobiography.record", "autobiography.discrepancy"),
             ("v0.28 autobiographical memory", "v0.32.2 autobiographical discrepancy")),
    SlotSpec("world_model", SlotKind.MECHANISM, "consequence / world model",
             ("world_model.prediction",), ("v0.25.2 consequence model",)),
    SlotSpec("skills", SlotKind.MECHANISM, "option / skill registry and selector",
             ("skills.selected_option",), ("v0.21-v0.22 goal options", "v0.24 episodic->goal->skill")),
    SlotSpec("planner", SlotKind.MECHANISM, "planner / replanner",
             ("planner.plan",), ("v0.25.2 mismatch-aware replanning",)),
    SlotSpec("goal_genesis", SlotKind.MECHANISM, "endogenous goal genesis",
             ("goal_genesis.active_goal",), ("v0.35.3 endogenous developmental goal genesis",)),
    SlotSpec("curriculum", SlotKind.MECHANISM, "recursive autobiographical curriculum hooks",
             ("curriculum.next_goal",), ("v0.34.1 event-consistent self-development", "v0.36.1 recursive curriculum")),
    SlotSpec("metacognition", SlotKind.MECHANISM, "metacognitive signal interface",
             ("metacognition.confidence",), ("v0.31 INVESTIGATE recruitment", "v0.7.1 UNKNOWN responses")),
    SlotSpec("persistence", SlotKind.INFRASTRUCTURE, "persistence / checkpoint manager"),
    SlotSpec("provenance", SlotKind.INFRASTRUCTURE, "experiment / provenance registry"),
    SlotSpec("coprocessor", SlotKind.INFRASTRUCTURE, "external coprocessor interface"),
)

SLOTS_BY_NAME = {s.name: s for s in REQUIRED_SLOTS}
MECHANISM_SLOTS = tuple(s.name for s in REQUIRED_SLOTS if s.kind == SlotKind.MECHANISM)
ENVIRONMENT_OWNER = "environment"          # owner of observation channels written by the runtime
DRIVER_OWNER = "driver"                    # owner of task-driver channels (a hash-pinned frozen evaluator driving the runtime)


def slot(name: str) -> SlotSpec:
    try:
        return SLOTS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown slot {name!r}; required slots: {[s.name for s in REQUIRED_SLOTS]}") from None
