"""Unified Individual integration-gate harness G1-G5 (draft gate UI-GATE-v0)."""

from .common import MechanismEvaluator, gate_record, prereg_unlocks
from .g1_retention import RetentionThreshold, run_g1
from .g2_interactions import STANDARD_INTERVENTIONS, InteractionMatrix, NamedIntervention, run_g2
from .g3_interference import STANDARD_SCHEDULES, InterferenceSchedule, run_g3
from .g4_persistence import run_g4_drift, run_g4_save_restore
from .g5_coprocessor import run_g5

__all__ = ["InteractionMatrix", "InterferenceSchedule", "MechanismEvaluator", "NamedIntervention",
           "RetentionThreshold", "STANDARD_INTERVENTIONS", "STANDARD_SCHEDULES", "gate_record", "prereg_unlocks",
           "run_g1", "run_g2", "run_g3", "run_g4_drift", "run_g4_save_restore", "run_g5"]
