"""Representation-neutral contracts shared by every track (Joint Contract §2.5).

Nothing in this package knows about spikes, GRUs, or legacy mechanisms. The
SNN (H1), the dense comparator (H2), the Unified Individual runtime, the
battery, and the integration-gate harness all depend on these contracts, so
neither experimental track can leak its implementation details into the
other or into the evaluators.

Dependencies: standard library and numpy only.
"""

from .checkpointable import Checkpointable, state_digest
from .features import FeatureSource, FeatureSpec
from .interventions import (InterventionError, InterventionKind, InterventionRecord, InterventionSpec,
                            InterventionTarget)
from .provenance import (ProvenanceRecord, SeedFamily, SeedPurpose, canonical_json, sha256_bytes,
                         sha256_file, sha256_json)
from .status import EvidenceStatus, ResultStatus, scientific_claim_allowed
from .streams import PHYSICAL_UNITS, ChannelKind, ChannelSpec, EmbodiedStream, Frame, StreamSpec
from .substrate import (BudgetReport, Capability, CapabilityUnavailable, GradientFirewallError,
                        LesionRecord, LesionSpec, PlasticityReport, Substrate, SubstrateFrozenError,
                        SubstrateMetadata, TrainReport, read_only)

__version__ = "0.38.1.dev0"

__all__ = [
    "BudgetReport", "Capability", "CapabilityUnavailable", "ChannelKind", "ChannelSpec", "Checkpointable",
    "EmbodiedStream", "EvidenceStatus", "FeatureSource", "FeatureSpec", "Frame", "GradientFirewallError",
    "InterventionError", "InterventionKind", "InterventionRecord", "InterventionSpec", "InterventionTarget",
    "LesionRecord", "LesionSpec", "PHYSICAL_UNITS", "PlasticityReport", "ProvenanceRecord", "ResultStatus",
    "SeedFamily", "SeedPurpose", "StreamSpec", "Substrate", "SubstrateFrozenError", "SubstrateMetadata",
    "TrainReport", "canonical_json", "read_only", "scientific_claim_allowed", "sha256_bytes", "sha256_file", "sha256_json",
    "state_digest",
]
