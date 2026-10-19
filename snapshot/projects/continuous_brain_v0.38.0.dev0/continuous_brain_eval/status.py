from __future__ import annotations
from dataclasses import dataclass,field
from enum import Enum
from typing import Any
class EvidenceStatus(str,Enum):
    ENGINEERING_ONLY="ENGINEERING_ONLY"
    UNREGISTERED_METRIC="UNREGISTERED_METRIC"
    REQUIRES_E2_BODY="REQUIRES_E2_BODY"
    NOT_RUN="NOT_RUN"
@dataclass
class EngineeringResult:
    item_id:str; status:EvidenceStatus; diagnostics:dict[str,Any]=field(default_factory=dict); notes:list[str]=field(default_factory=list)
    scientific_interpretation_permitted:bool=False
    def __post_init__(self):
        if self.scientific_interpretation_permitted: raise ValueError('v0.38.0.dev0 engineering results may not authorize scientific interpretation')
