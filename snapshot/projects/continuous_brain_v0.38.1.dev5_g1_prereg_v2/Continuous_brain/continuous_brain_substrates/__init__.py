"""Substrates behind the common API: the E0 SNN adapter (H1) and the DENSE-RNN comparator (H2).

Track-neutral pieces (the shared encoder, budget matching) live here too, so
neither track imports the other.
"""

from __future__ import annotations

from dataclasses import replace

from .dense_rnn import (BudgetMatchError, DenseRNNConfig, DenseRNNSubstrate, SingleExposureViolation,
                        gru_parameter_count, gru_recurrent_weight_count, solve_hidden_size,
                        solve_hidden_size_recurrent)
from .encoders import EncoderConfig, SensoryEncoder, motor_stream_spec, sensory_stream_spec
from .snn_adapter import SNNAdapterConfig, SNNSubstrate

__version__ = "0.38.1.dev3"


def build_matched_pair(e0_config, dense: DenseRNNConfig | None = None, frame_dt_s: float = 0.01,
                       snn_adapter: SNNAdapterConfig | None = None, device: str = "cpu"):
    """An SNN substrate and a GRU whose trainable parameters match its active synapses (+-10 %)."""
    from continuous_brain_snn.engine import SNNEngine
    snn = SNNSubstrate(SNNEngine(e0_config, device=device), snn_adapter, frame_dt_s=frame_dt_s)
    target = snn.budget().active_synapses
    dense = replace(dense or DenseRNNConfig(seed=e0_config.seed), target_learned_scalars=target,
                    hidden_size=None, frame_dt_s=frame_dt_s, device=device)
    return snn, DenseRNNSubstrate(snn.encoder_cfg, dense)


def build_a1b_comparators(e0_config, frame_dt_s: float = 0.01, snn_adapter: SNNAdapterConfig | None = None,
                          device: str = "cpu"):
    """A1.1b: return SNN, literal total-match GRU, and recurrent-core-match GRU.

    DENSE-TOTAL preserves the original A1.1 interpretation. DENSE-CORE matches
    GRUCell.weight_hh scalars to cortex->cortex active SNN synapses. Both are
    report-only until the amended preregistration is approved.
    """
    from continuous_brain_snn.engine import SNNEngine
    snn = SNNSubstrate(SNNEngine(e0_config, device=device), snn_adapter, frame_dt_s=frame_dt_s)
    b = snn.budget()
    total_cfg = DenseRNNConfig(seed=e0_config.seed, target_learned_scalars=b.active_synapses, budget_mode="total",
                               frame_dt_s=frame_dt_s, device=device)
    core_target = int(b.notes["cortex_to_cortex_active_synapses"])
    core_cfg = DenseRNNConfig(seed=e0_config.seed, target_recurrent_scalars=core_target,
                              budget_mode="recurrent_core", frame_dt_s=frame_dt_s, device=device)
    return snn, DenseRNNSubstrate(snn.encoder_cfg, total_cfg), DenseRNNSubstrate(snn.encoder_cfg, core_cfg)


__all__ = ["BudgetMatchError", "DenseRNNConfig", "DenseRNNSubstrate", "EncoderConfig", "SNNAdapterConfig",
           "SNNSubstrate", "SensoryEncoder", "SingleExposureViolation", "build_matched_pair",
           "gru_parameter_count", "gru_recurrent_weight_count", "motor_stream_spec", "sensory_stream_spec",
           "solve_hidden_size", "solve_hidden_size_recurrent", "build_a1b_comparators"]
