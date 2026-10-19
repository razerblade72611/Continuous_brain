from __future__ import annotations
from dataclasses import dataclass,field
from typing import Iterable
from continuous_brain_unified.interventions import PredictionCell,InterventionSpec
from continuous_brain_unified.runtime import UnifiedRuntime
from .status import EvidenceStatus,EngineeringResult

@dataclass
class IntegrationPlan:
    interventions:list[InterventionSpec]=field(default_factory=list)
    predictions:list[PredictionCell]=field(default_factory=list)
    interference_schedules:list[str]=field(default_factory=list)
    long_run_duration:str|None=None

def validate_prediction_matrix(plan:IntegrationPlan,mechanism_ids:Iterable[str])->list[str]:
    errors=[]; cells={(c.intervention_id,c.mechanism_id):c for c in plan.predictions}
    for i in plan.interventions:
        for m in mechanism_ids:
            c=cells.get((i.intervention_id,m))
            if c is None: errors.append(f'missing prediction: {i.intervention_id} x {m}')
            elif not c.rationale.strip(): errors.append(f'missing rationale: {i.intervention_id} x {m}')
    return errors

class IntegrationGateHarness:
    def g1_binding_inventory(self,runtime:UnifiedRuntime)->EngineeringResult:
        s=runtime.status(); return EngineeringResult('G1',EvidenceStatus.ENGINEERING_ONLY,{'components':s.components,'unbound':s.unbound},['binding inventory only; frozen evaluators are not automatically rerun'])
    def g2_plan(self,plan:IntegrationPlan,mechanism_ids:list[str])->EngineeringResult:
        errs=validate_prediction_matrix(plan,mechanism_ids); return EngineeringResult('G2',EvidenceStatus.ENGINEERING_ONLY,{'plan_complete':not errs,'errors':errs},['plan validation only'])
    def g3_plan(self,plan:IntegrationPlan)->EngineeringResult:
        return EngineeringResult('G3',EvidenceStatus.ENGINEERING_ONLY,{'schedules':plan.interference_schedules},['no degradation margin is preregistered here'])
    def g4_plan(self,plan:IntegrationPlan)->EngineeringResult:
        return EngineeringResult('G4',EvidenceStatus.ENGINEERING_ONLY,{'long_run_duration':plan.long_run_duration},['shell checkpointing does not imply legacy-state preservation'])
    def g5_coprocessor_ablation(self,runtime:UnifiedRuntime)->EngineeringResult:
        return EngineeringResult('G5',EvidenceStatus.ENGINEERING_ONLY,{'coprocessor_enabled':runtime.status().coprocessor_enabled},['ablation plumbing only'])
