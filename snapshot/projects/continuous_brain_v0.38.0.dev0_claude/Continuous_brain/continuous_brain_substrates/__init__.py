"""Substrates behind the common API: the E0 SNN adapter (H1) and the DENSE-RNN comparator (H2).

Track-neutral pieces (the shared encoder, budget matching) live here too, so
neither track imports the other.
"""

from __future__ import annotations

from dataclasses import replace

from .dense_rnn import (BudgetMatchError, DenseRNNConfig, DenseRNNSubstrate, SingleExposureViolation,
                        gru_parameter_count, solve_hidden_size)
from .encoders import EncoderConfig, SensoryEncoder, motor_stream_spec, sensory_stream_spec
from .snn_adapter import SNNAdapterConfig, SNNSubstrate

__version__ = "0.38.0.dev0"


def build_matched_pair(e0_config, dense: DenseRNNConfig | None = None, frame_dt_s: float = 0.01,
                       snn_adapter: SNNAdapterConfig | None = None, device: str = "cpu"):
    """An SNN substrate and a GRU whose trainable parameters match its active synapses (+-10 %)."""
    from continuous_brain_snn.engine import SNNEngine
    snn = SNNSubstrate(SNNEngine(e0_config, device=device), snn_adapter, frame_dt_s=frame_dt_s)
    target = snn.budget().active_synapses
    dense = replace(dense or DenseRNNConfig(seed=e0_config.seed), target_learned_scalars=target,
                    hidden_size=None, frame_dt_s=frame_dt_s, device=device)
    return snn, DenseRNNSubstrate(snn.encoder_cfg, dense)


__all__ = ["BudgetMatchError", "DenseRNNConfig", "DenseRNNSubstrate", "EncoderConfig", "SNNAdapterConfig",
           "SNNSubstrate", "SensoryEncoder", "SingleExposureViolation", "build_matched_pair",
           "gru_parameter_count", "motor_stream_spec", "sensory_stream_spec", "solve_hidden_size"]
