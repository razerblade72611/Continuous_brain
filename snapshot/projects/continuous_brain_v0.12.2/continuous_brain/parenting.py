"""Parent/teacher registry and gradual-weaning policy for Continuous Brain v0.9.

Parents are replaceable external teachers.  They are explicitly not part of the
persistent individual checkpoint.  Student cortices can learn from their outputs,
but runtime authority is reduced only after held-out competency gates are met.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Iterable

CAPABILITIES = (
    "vision",
    "language",
    "temporal",
    "speech",
    "motor",
    "values",
    "reasoning",
)


@dataclass(frozen=True)
class ParentSpec:
    parent_id: str
    capability: str
    model: str
    interface: str
    role: str = "teacher"
    endpoint: str | None = None
    notes: str = ""

    def __post_init__(self):
        if self.capability not in CAPABILITIES:
            raise ValueError(f"Unknown parent capability: {self.capability}")
        if not self.parent_id or not self.model or not self.interface:
            raise ValueError("parent_id, model, and interface are required")


DEFAULT_PARENTS = (
    ParentSpec(
        "vision.dinov3.base",
        "vision",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "huggingface-image-embedding",
        notes="Balanced spatial-vision parent; student learns its own raw-pixel visual cortex.",
    ),
    ParentSpec(
        "language.qwen3_5.4b",
        "language",
        "Qwen/Qwen3.5-4B",
        "openai-chat",
        notes="Default language/reasoning parent. Upgradeable to 9B or stronger service models.",
    ),
    ParentSpec(
        "temporal.vjepa2",
        "temporal",
        "facebook/vjepa2-vitl-fpc64-256",
        "embedding-service",
        notes="Temporal/world-understanding parent boundary; optional during early local proof.",
    ),
    ParentSpec(
        "speech.whisper.large_v3_turbo",
        "speech",
        "openai/whisper-large-v3-turbo",
        "huggingface-asr",
        notes="Speech-recognition parent; the infant audio cortex is trained separately.",
    ),
    ParentSpec(
        "reasoning.qwen3_5.35b_a3b",
        "reasoning",
        "Qwen/Qwen3.5-35B-A3B-FP8",
        "openai-chat",
        notes="Optional cloud parent for difficult curriculum generation/evaluation, not a runtime dependency.",
    ),
    ParentSpec(
        "motor.foraging.scripted",
        "motor",
        "continuous_brain.foraging_parent_action",
        "internal-demonstration-policy",
        notes="Bounded nursery motor demonstrator. Replaceable by stronger embodied parents later.",
    ),
    ParentSpec(
        "values.care.curriculum",
        "values",
        "continuous_brain.care_curriculum",
        "structured-developmental-targets",
        notes="Inspectable developmental value/care targets; competence still requires diverse held-out evidence.",
    ),
)


@dataclass
class CapabilityWeaning:
    capability: str
    parent_authority: float = 1.0
    student_authority: float = 0.0
    best_score: float = 0.0
    consecutive_passes: int = 0
    graduated: bool = False
    last_updated: float = field(default_factory=time.time)

    def observe(self, score: float, *, threshold: float = 0.85, required_passes: int = 3,
                step: float = 0.20, min_parent: float = 0.0) -> None:
        score = float(max(0.0, min(1.0, score)))
        self.best_score = max(self.best_score, score)
        self.consecutive_passes = self.consecutive_passes + 1 if score >= threshold else 0
        if self.consecutive_passes >= required_passes:
            self.parent_authority = max(float(min_parent), self.parent_authority - float(step))
            self.student_authority = 1.0 - self.parent_authority
            self.consecutive_passes = 0
        self.graduated = self.parent_authority <= min_parent + 1e-9 and score >= threshold
        self.last_updated = time.time()


class ParentRegistry:
    """Persistent, auditable parent catalog and weaning state.

    Parent replacements never mutate identity state.  The history makes it possible
    to know which external models taught which developmental period.
    """

    FORMAT = "continuous_brain.parent_registry.v090"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.parents: dict[str, ParentSpec] = {}
        self.active: dict[str, str] = {}
        self.weaning: dict[str, CapabilityWeaning] = {
            cap: CapabilityWeaning(cap) for cap in CAPABILITIES
        }
        self.history: list[dict] = []
        if self.path.exists():
            self.load()
        else:
            for spec in DEFAULT_PARENTS:
                self.register(spec, activate=True, event="default_parent")
            self.save()

    def register(self, spec: ParentSpec, *, activate: bool = False, event="registered") -> None:
        self.parents[spec.parent_id] = spec
        if activate:
            previous = self.active.get(spec.capability)
            self.active[spec.capability] = spec.parent_id
            self.history.append({
                "time": time.time(), "event": event, "capability": spec.capability,
                "parent_id": spec.parent_id, "replaced": previous,
            })

    def active_parent(self, capability: str) -> ParentSpec | None:
        parent_id = self.active.get(capability)
        return self.parents.get(parent_id) if parent_id else None

    def upgrade(self, spec: ParentSpec) -> None:
        """Activate a new parent while preserving child authority and old developmental history.

        Competency streaks are parent-specific evidence, so a new parent resets the
        score/pass streak but does not make an already independent child dependent again.
        """
        self.register(spec, activate=True, event="parent_upgrade")
        state = self.weaning[spec.capability]
        state.best_score = 0.0
        state.consecutive_passes = 0
        state.graduated = False
        state.last_updated = time.time()
        self.save()

    def record_score(self, capability: str, score: float, **kwargs) -> CapabilityWeaning:
        if capability not in self.weaning:
            raise ValueError(capability)
        self.weaning[capability].observe(score, **kwargs)
        self.history.append({
            "time": time.time(), "event": "competency_score", "capability": capability,
            "score": float(score), "parent_authority": self.weaning[capability].parent_authority,
        })
        self.save()
        return self.weaning[capability]

    def guidance_probability(self, capability: str) -> float:
        return float(self.weaning[capability].parent_authority)

    def to_dict(self) -> dict:
        return {
            "format": self.FORMAT,
            "parents": {k: asdict(v) for k, v in self.parents.items()},
            "active": dict(self.active),
            "weaning": {k: asdict(v) for k, v in self.weaning.items()},
            "history": list(self.history),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.replace(self.path)

    def load(self) -> None:
        data = json.loads(self.path.read_text())
        if data.get("format") != self.FORMAT:
            raise ValueError("Unsupported parent registry format")
        self.parents = {k: ParentSpec(**v) for k, v in data.get("parents", {}).items()}
        self.active = dict(data.get("active", {}))
        # Forward-compatible registry migration: newly formalized parent roles are
        # added without replacing any already-active parent for that capability.
        for spec in DEFAULT_PARENTS:
            self.parents.setdefault(spec.parent_id, spec)
            self.active.setdefault(spec.capability, spec.parent_id)
        self.weaning = {
            cap: CapabilityWeaning(**data.get("weaning", {}).get(cap, {"capability": cap}))
            for cap in CAPABILITIES
        }
        self.history = list(data.get("history", []))


def parent_summary(registry: ParentRegistry) -> list[dict]:
    out = []
    for capability in CAPABILITIES:
        spec = registry.active_parent(capability)
        state = registry.weaning[capability]
        out.append({
            "capability": capability,
            "parent": spec.model if spec else None,
            "parent_id": spec.parent_id if spec else None,
            "parent_authority": state.parent_authority,
            "student_authority": state.student_authority,
            "best_score": state.best_score,
            "graduated": state.graduated,
        })
    return out
