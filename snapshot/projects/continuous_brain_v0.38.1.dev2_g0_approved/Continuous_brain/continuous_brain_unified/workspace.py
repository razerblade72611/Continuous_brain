"""Interfaces for future workspace (GWT-2/GWT-3) and attention-schema (AST-1) mechanisms.

C4: v0.38.0 ships the contracts and a registration point only. There is no
implementation here, disabled or otherwise. The first mechanism arrives with
its own APPROVED preregistration, and `register_experimental` refuses anything
else. Historical failed broadcast results (v0.29) stay failed; nothing here
touches them.

What a workspace mechanism must expose, so G2 interventions and the
indicator registry can test it without knowing its internals:
  * a capacity limit (GWT-2): at most `capacity` items admitted per tick
  * the competition (which candidates, which won, why) as channel data
  * the broadcast (GWT-3): the admitted content, written to `workspace.state`,
    readable by every other bound component
An attention schema (AST-1) must write a model of the current attention state
and a prediction of its next state, and accept control through a channel.
"""

from __future__ import annotations

import abc

from .components import Component, ComponentProvenance, ProvenanceKind
from .prereg import Preregistration, PreregistrationRegistry


class WorkspaceMechanism(Component, abc.ABC):
    slot = "workspace"

    @property
    @abc.abstractmethod
    def capacity(self) -> int:
        """Maximum number of items admitted to the workspace per tick."""

    @property
    @abc.abstractmethod
    def candidate_channels(self) -> tuple[str, ...]:
        """Channels whose contents compete for admission."""


class AttentionSchemaMechanism(Component, abc.ABC):
    slot = "metacognition"

    @property
    @abc.abstractmethod
    def attention_state_channel(self) -> str:
        """Channel carrying the modelled attention state."""

    @property
    @abc.abstractmethod
    def attention_prediction_channel(self) -> str:
        """Channel carrying the predicted next attention state."""


_REGISTERED: dict[type, Preregistration] = {}


def register_experimental(cls: type, prereg_id: str, registry: PreregistrationRegistry | None = None) -> type:
    """Admit a new mechanism class under an APPROVED preregistration that covers its slot."""
    if not (isinstance(cls, type) and issubclass(cls, Component)):
        raise TypeError("only Component subclasses can be registered")
    reg = registry or PreregistrationRegistry.load()
    _REGISTERED[cls] = reg.require(prereg_id, getattr(cls, "slot", None))
    return cls


def experimental_provenance(component: Component) -> ComponentProvenance:
    prereg = _REGISTERED.get(type(component))
    if prereg is None:
        raise PermissionError(f"{type(component).__name__} is not registered under an approved preregistration")
    return ComponentProvenance(ProvenanceKind.EXPERIMENTAL_PREREG, {"prereg_id": prereg.id,
                                                                    "class": type(component).__qualname__})


def registered_prereg(cls: type) -> Preregistration | None:
    return _REGISTERED.get(cls)
