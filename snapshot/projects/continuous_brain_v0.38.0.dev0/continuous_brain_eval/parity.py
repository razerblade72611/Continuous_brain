from __future__ import annotations
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class RateParity:
    rate_a:float; rate_b:float; n_neurons:int; duration_s:float; tolerance:float; difference:float; underpowered:bool; within_tolerance:bool

def power_aware_rate_parity(rate_a:float,rate_b:float,n_neurons:int,duration_s:float,z:float=3.57,relative_floor:float=0.05)->RateParity:
    if n_neurons<=0 or duration_s<=0: raise ValueError('n_neurons and duration_s must be positive')
    pooled=max((rate_a+rate_b)/2.0,1e-12); se=math.sqrt(max(rate_a+rate_b,0.0)/(n_neurons*duration_s)); sampling=z*se; relative=relative_floor*pooled; tol=max(relative,sampling); diff=abs(rate_a-rate_b)
    return RateParity(rate_a,rate_b,n_neurons,duration_s,tol,diff,sampling>relative,diff<=tol)

def legacy_relative_rate_check(rate_a:float,rate_b:float,tolerance:float=0.05)->bool:
    denom=max(abs(rate_a),abs(rate_b),1e-12); return abs(rate_a-rate_b)/denom <= tolerance
