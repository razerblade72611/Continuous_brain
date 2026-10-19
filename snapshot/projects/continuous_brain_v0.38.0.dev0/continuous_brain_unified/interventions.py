from __future__ import annotations
from dataclasses import dataclass,field
from enum import Enum
from typing import Any,Callable

class ExpectedEffect(str,Enum): CHANGE="CHANGE"; NO_CHANGE="NO_CHANGE"; INCREASE="INCREASE"; DECREASE="DECREASE"
@dataclass(frozen=True)
class InterventionSpec:
    intervention_id:str; target_component:str; operation:str; parameters:dict[str,Any]=field(default_factory=dict)
@dataclass(frozen=True)
class PredictionCell:
    intervention_id:str; mechanism_id:str; expected:ExpectedEffect; rationale:str
class InterventionRegistry:
    def __init__(self): self._handlers:dict[tuple[str,str],Callable[...,Any]]={}
    def register(self,target:str,operation:str,handler:Callable[...,Any])->None: self._handlers[(target,operation)]=handler
    def apply(self,spec:InterventionSpec)->Any:
        h=self._handlers.get((spec.target_component,spec.operation))
        if h is None: raise KeyError(f'unbound intervention handler: {(spec.target_component,spec.operation)}')
        return h(**spec.parameters)
