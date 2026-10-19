"""Unified Individual composition runtime (Joint Contract §1; C1, C3, C4).

Named `continuous_brain_unified`, never `continuous_brain`: that name belongs
to the frozen legacy package, and installing over it would shadow the
originals (C3). Version 0.38.0.dev0 is a pre-release marker until DAV-76
settles naming and something passes a gate.
"""

__version__ = "0.38.1.dev7"

from .channels import ChannelDecl, ChannelError, ChannelStore, ChannelWriteError  # noqa: E402
from .components import Component, ComponentProvenance, ProvenanceKind, TickContext  # noqa: E402
from .coprocessor import (Coprocessor, CoprocessorBudgetExceeded, CoprocessorClient, CoprocessorRequest,  # noqa: E402
                          CoprocessorResponse, NullCoprocessor)
from .interventions import InterventionManager  # noqa: E402
from .runtime import (BindingError, BindingPolicy, ReadOnlyRuntimeView, RuntimeConfig, SlotReport, TickResult,  # noqa: E402
                      UnifiedRuntime)
from .scheduler import SchedulingError, schedule  # noqa: E402
from .slots import MECHANISM_SLOTS, REQUIRED_SLOTS, SlotKind, SlotSpec  # noqa: E402

__all__ = ["BindingError", "BindingPolicy", "ChannelDecl", "ChannelError", "ChannelStore", "ChannelWriteError",
           "Component", "ComponentProvenance", "Coprocessor", "CoprocessorBudgetExceeded", "CoprocessorClient",
           "CoprocessorRequest", "CoprocessorResponse", "InterventionManager", "MECHANISM_SLOTS", "NullCoprocessor",
           "ProvenanceKind", "REQUIRED_SLOTS", "ReadOnlyRuntimeView", "RuntimeConfig", "SchedulingError",
           "SlotKind", "SlotReport", "SlotSpec", "TickContext", "TickResult", "UnifiedRuntime", "schedule"]
