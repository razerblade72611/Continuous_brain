from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .goals import canonical_name, parse_goal
from .models import ActionKind, WorldState


@dataclass(frozen=True)
class TaskBaseline:
    """Minimal simulator evidence captured when a user task becomes active."""

    held_object_id: str | None = None
    discovery_region: str | None = None


@dataclass(frozen=True)
class TaskContract:
    """Deterministic, user-task-derived mechanical contract.

    A contract never creates an autonomous preference. It only describes a concrete
    operation already present in the active user task and the simulator evidence
    required before that task may be marked complete.
    """

    raw: str
    kind: str = "Unknown"
    target: str | None = None
    source: str | None = None
    recipient: str | None = None
    relation: str | None = None

    @property
    def is_known(self) -> bool:
        return self.kind != "Unknown"

    @property
    def is_text(self) -> bool:
        return self.kind == "TextPeer"

    @property
    def is_interaction(self) -> bool:
        return self.kind in {
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
            ActionKind.TOGGLE_ON.value,
            ActionKind.TOGGLE_OFF.value,
            "DropObject",
        }

    @property
    def is_world_verifiable(self) -> bool:
        return self.is_interaction or self.kind == "ExitRegion"

    def for_llm(self) -> dict | None:
        """Safe contract summary derived only from the user's task text.

        No object IDs, coordinates, hidden object catalog, or hidden state is exposed.
        """
        if not self.is_known:
            return None
        if self.kind == "TextPeer":
            return {
                "required_capability": "TextPeer",
                "recipient": self.recipient,
                "completion_evidence": "an actual delivered TextPeer event",
                "note": "message wording remains your choice; say/status text is not a peer message",
            }
        if self.kind == "ExitRegion":
            return {
                "required_outcome": "physically leave the starting room/region",
                "completion_evidence": "simulator position enters a different connected Discovery region",
                "note": "the private motor layer may select and execute a physical doorway route; no teleport is used",
            }
        return {
            "requested_interaction": self.kind,
            "target_phrase": self.target,
            "source_phrase": self.source,
            "completion_evidence": "post-action simulator state",
            "note": "Approach/Face are preparation only; once the requested interaction is mechanically available, use it",
        }


def _clean_phrase(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    text = re.sub(r"\b(?:please|now)\b[.!?]*$", "", text, flags=re.IGNORECASE).strip(" .!?\t")
    return text or None


def _peer_labels(social_context: dict | None) -> list[str]:
    if not social_context:
        return []
    labels: list[str] = []
    for row in social_context.get("peer_identity") or []:
        if isinstance(row, dict):
            label = str(row.get("label") or "").strip()
            if label:
                labels.append(label)
    return labels


def _resolve_peer_token(token: str | None, peer_labels: Iterable[str]) -> str | None:
    if not token:
        return None
    wanted = canonical_name(token)
    for label in peer_labels:
        if canonical_name(label) == wanted:
            return str(label)
    return None


def parse_task_contract(
    task: str | None,
    *,
    self_label: str | None = None,
    peer_labels: Iterable[str] = (),
) -> TaskContract:
    raw = " ".join(str(task or "").strip().split())
    if not raw:
        return TaskContract(raw="")

    peers = [str(label) for label in peer_labels if str(label).strip()]
    low = raw.lower()

    # Explicit peer messaging. "each other" is resolved only when there is exactly
    # one other known peer, preserving the user's routing semantics without guessing.
    if re.search(r"\b(?:text|message|messaging)\b", low):
        recipient: str | None = None
        if re.search(r"\b(?:each other|one another)\b", low) and len(peers) == 1:
            recipient = peers[0]
        else:
            patterns = [
                r"\b(?:text|message)\s+(?:agent\s+)?(?P<peer>[a-zA-Z0-9_-]+)\b",
                r"\bsend\s+(?:a\s+)?(?:text|message)\s+to\s+(?:agent\s+)?(?P<peer>[a-zA-Z0-9_-]+)\b",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw, flags=re.IGNORECASE)
                if match:
                    token = match.group("peer")
                    if canonical_name(token) not in {"each", "one", "the", "another"}:
                        recipient = _resolve_peer_token(token, peers) or _clean_phrase(token)
                        break
        if recipient is not None or re.search(r"\b(?:each other|one another)\b", low):
            return TaskContract(raw=raw, kind="TextPeer", recipient=recipient)

    spec = parse_goal(raw)
    # Composite placement is staged by next_required_interaction().
    if spec.has_put:
        return TaskContract(
            raw=raw,
            kind=ActionKind.PUT.value,
            source=_clean_phrase(spec.source_object),
            target=_clean_phrase(spec.put_target),
            relation=spec.put_relation,
        )
    if spec.pickup_object:
        return TaskContract(raw=raw, kind=ActionKind.PICKUP.value, target=_clean_phrase(spec.pickup_object))
    if spec.open_object:
        return TaskContract(raw=raw, kind=ActionKind.OPEN.value, target=_clean_phrase(spec.open_object))
    if spec.close_object:
        return TaskContract(raw=raw, kind=ActionKind.CLOSE.value, target=_clean_phrase(spec.close_object))

    toggle_on = re.fullmatch(
        r"(?:turn|switch|toggle)\s+(?:the\s+)?(?P<object>[a-zA-Z0-9_ -]+?)\s+on[.!?]?",
        raw,
        flags=re.IGNORECASE,
    )
    if not toggle_on:
        toggle_on = re.fullmatch(
            r"(?:turn|switch)\s+on\s+(?:the\s+)?(?P<object>[a-zA-Z0-9_ -]+?)[.!?]?",
            raw,
            flags=re.IGNORECASE,
        )
    if toggle_on:
        return TaskContract(raw=raw, kind=ActionKind.TOGGLE_ON.value, target=_clean_phrase(toggle_on.group("object")))

    toggle_off = re.fullmatch(
        r"(?:turn|switch|toggle)\s+(?:the\s+)?(?P<object>[a-zA-Z0-9_ -]+?)\s+off[.!?]?",
        raw,
        flags=re.IGNORECASE,
    )
    if not toggle_off:
        toggle_off = re.fullmatch(
            r"(?:turn|switch)\s+off\s+(?:the\s+)?(?P<object>[a-zA-Z0-9_ -]+?)[.!?]?",
            raw,
            flags=re.IGNORECASE,
        )
    if toggle_off:
        return TaskContract(raw=raw, kind=ActionKind.TOGGLE_OFF.value, target=_clean_phrase(toggle_off.group("object")))

    drop = re.fullmatch(
        r"(?:drop|release)\s+(?:(?:the\s+)?(?P<object>[a-zA-Z0-9_ -]+?))?[.!?]?",
        raw,
        flags=re.IGNORECASE,
    )
    if drop:
        return TaskContract(raw=raw, kind="DropObject", target=_clean_phrase(drop.group("object")))

    # Exit/escape is only considered verifiable in the custom Discovery world,
    # where connected regions have known physical boundaries. It never completes on
    # an LLM claim alone.
    if re.search(r"\b(?:escape|exit|leave|get\s+out\s+of)\b", low):
        return TaskContract(raw=raw, kind="ExitRegion", target="kitchen" if "kitchen" in low else None)

    return TaskContract(raw=raw)


def contract_from_context(task: str | None, social_context: dict | None = None, self_label: str | None = None) -> TaskContract:
    return parse_task_contract(
        task,
        self_label=self_label,
        peer_labels=_peer_labels(social_context),
    )


def discovery_region(world: WorldState) -> str | None:
    """Classify the v0.9 Discovery scene by physically crossed doorway thresholds."""
    if "discovery" not in str(world.scene or "").lower():
        return None
    x = float(world.pose.position.get("x", 0.0))
    z = float(world.pose.position.get("z", 0.0))
    # Door centers are approximately east x=2.4 and north z=2.5. Thresholds are
    # deliberately beyond the leaf/threshold so standing at a closed door is not exit.
    if x >= 2.75:
        return "east_experiment"
    if z >= 2.85:
        return "north_archive"
    return "kitchen"


def capture_task_baseline(world: WorldState) -> TaskBaseline:
    return TaskBaseline(
        held_object_id=world.held_object_id,
        discovery_region=discovery_region(world),
    )


def _matches(obj, phrase: str | None) -> bool:
    if not phrase:
        return False
    needle = canonical_name(phrase)
    return needle in {canonical_name(obj.object_type), canonical_name(obj.name)} or (
        needle and (needle in canonical_name(obj.object_type) or needle in canonical_name(obj.name))
    )


def contract_satisfied(contract: TaskContract, world: WorldState, baseline: TaskBaseline | None = None) -> bool:
    if contract.kind == ActionKind.PICKUP.value:
        return any(obj.is_picked_up for obj in world.find(contract.target or ""))

    if contract.kind == ActionKind.PUT.value:
        if not contract.target or not contract.source:
            return False
        sources = world.find(contract.source)
        targets = world.find(contract.target)
        target_ids = {obj.object_id for obj in targets}
        return bool(target_ids) and any(
            any(parent in target_ids for parent in source.parent_receptacles)
            for source in sources
        )

    if contract.kind == ActionKind.OPEN.value:
        return any(obj.is_open is True for obj in world.find(contract.target or ""))

    if contract.kind == ActionKind.CLOSE.value:
        return any(obj.is_open is False for obj in world.find(contract.target or ""))

    if contract.kind == ActionKind.TOGGLE_ON.value:
        return any(obj.is_toggled is True for obj in world.find(contract.target or ""))

    if contract.kind == ActionKind.TOGGLE_OFF.value:
        return any(obj.is_toggled is False for obj in world.find(contract.target or ""))

    if contract.kind == "DropObject":
        return bool(baseline and baseline.held_object_id and world.held_object_id is None)

    if contract.kind == "ExitRegion":
        before = None if baseline is None else baseline.discovery_region
        after = discovery_region(world)
        if before is None or after is None:
            return False
        if contract.target and canonical_name(contract.target) == "kitchen":
            return before == "kitchen" and after != "kitchen"
        return after != before

    return False


def next_required_interaction(contract: TaskContract, world: WorldState) -> tuple[str, str | None] | None:
    """Return the next explicit user-authorized motor operation, if any."""
    if not contract.is_interaction:
        return None
    if contract.kind == ActionKind.PUT.value:
        # Placement requires holding the source first. This is deterministic task
        # mechanics, not an autonomous preference.
        if world.held_object_id is None:
            if contract.source:
                return ActionKind.PICKUP.value, contract.source
            return None
        return ActionKind.PUT.value, contract.target
    if contract.kind == "DropObject":
        return ("DropObject", None) if world.held_object_id is not None else None
    return contract.kind, contract.target


def semantic_reference_matches(a: str | None, b: str | None) -> bool:
    ca = canonical_name(a)
    cb = canonical_name(b)
    if not ca or not cb:
        return False
    return ca == cb or ca in cb or cb in ca
