"""Porting template for the canonical core: InfantBrain / CorticalNucleus v0.11.3.

The real InfantBrain source and its 14,000-update checkpoint live on the home
machine (WSL: ~/projects/continuous_brain_v0.12.2, canonical lineage v0.11.3).
They were not reachable when this repo was populated, so this adapter is a
template, not an implementation. It refuses to construct so nothing can run
on it by mistake.

To port (from the home machine):
  1. Snapshot first: scripts/snapshot.sh <checkpoint> pre-port
  2. Copy the v0.11.3 source into continuous_brain/core/infantbrain/ unchanged.
  3. Implement each method below by delegating to InfantBrain:
       step              -> run its integration loop for duration_s; export only
                            the designated projection neurons' spikes as
                            SpikeEvents with absolute float64 times
       reset_state       -> clear membranes/synaptic state/eligibility, keep weights
       set_plasticity_tracking / apply_plasticity
                         -> its local learning rule, gated by the modulator;
                            no weight change unless apply_plasticity is called
       state_dict / load_state_dict
                         -> must round-trip the 14,000-update checkpoint exactly
       projection_regions -> its stream/region label per projection neuron, so
                            populations follow its real anatomy
  4. Wire the working-memory input (fed zeros to date) to real input neurons
     before this core gates anything (spec: "InfantBrain's role").
  5. Run tests/test_core_contract.py against the adapter.
"""

from __future__ import annotations

from .interface import SpikingCore


class InfantBrainAdapter(SpikingCore):  # pragma: no cover - template only
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "InfantBrain v0.11.3 has not been ported into this repo yet. "
            "See the porting steps in continuous_brain/core/infantbrain_adapter.py. "
            "Use ReferenceLIFCore only for testing the surrounding system.")
