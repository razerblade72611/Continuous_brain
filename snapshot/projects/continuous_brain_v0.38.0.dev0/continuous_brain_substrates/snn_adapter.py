from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from .contracts import BudgetReport, ComputeReport, GradientFirewallError, SubstrateState, TrainingBatch, UnsupportedOperation

class ExternalSNNAdapter:
    """Adapter over a real external E0 engine. It never supplies a stand-in engine."""
    def __init__(self, engine: Any, *, name: str="E0-SNN/external"):
        if engine is None: raise ValueError("a real external E0 engine is required")
        self.engine=engine; self._name=name; self._timestamp=0.0
        self._inference_steps=self._plasticity_steps=0
        self.assert_gradient_firewall()

    @property
    def name(self)->str: return self._name

    def _tensors(self):
        if hasattr(self.engine,"parameters") and callable(self.engine.parameters):
            try:
                yield from self.engine.parameters()
            except TypeError: pass
        for v in getattr(self.engine,"__dict__",{}).values():
            if isinstance(v,torch.Tensor): yield v

    def assert_gradient_firewall(self)->None:
        bad=[t for t in self._tensors() if getattr(t,"requires_grad",False)]
        if bad: raise GradientFirewallError(f"SNN gradient firewall violation: {len(bad)} tensor(s) require grad")

    def reset(self, seed:int|None=None)->None:
        self.assert_gradient_firewall()
        if hasattr(self.engine,"reset"):
            try: self.engine.reset(seed=seed)
            except TypeError: self.engine.reset()
        else: raise UnsupportedOperation("external E0 engine has no reset()")
        self._timestamp=0.0

    def step(self, observation:torch.Tensor, dt:float)->SubstrateState:
        self.assert_gradient_firewall()
        if not hasattr(self.engine,"step"): raise UnsupportedOperation("external E0 engine has no step()")
        out=self.engine.step(observation, dt) if _accepts_dt(self.engine.step) else self.engine.step(observation)
        self._timestamp += float(dt); self._inference_steps += 1
        return self._state_from(out)

    def _state_from(self,out:Any=None)->SubstrateState:
        if isinstance(out,SubstrateState): return out
        if out is None and hasattr(self.engine,"expose_state"): out=self.engine.expose_state()
        if out is None and hasattr(self.engine,"state"): out=self.engine.state
        if isinstance(out,dict):
            for k in ("vector","state","spikes","features"):
                if k in out: return SubstrateState(torch.as_tensor(out[k]).detach().cpu().flatten(), self._timestamp, {"feature_kind":k})
        if isinstance(out,torch.Tensor): return SubstrateState(out.detach().cpu().flatten(),self._timestamp,{"feature_kind":"external"})
        raise UnsupportedOperation("cannot expose a vector state from external E0 engine")

    def expose_state(self)->SubstrateState: return self._state_from(None)

    def plasticity_step(self, modulators:Mapping[str,float]|None=None)->None:
        self.assert_gradient_firewall()
        for n in ("plasticity_step","local_plasticity_step","learn_local"):
            f=getattr(self.engine,n,None)
            if callable(f):
                f(modulators or {}); self._plasticity_steps+=1; self.assert_gradient_firewall(); return
        raise UnsupportedOperation("external E0 engine exposes no recognized local-plasticity method")

    def train_step(self,batch:TrainingBatch):
        raise GradientFirewallError("gradient training is forbidden for the SNN substrate")

    def freeze(self)->None:
        f=getattr(self.engine,"freeze",None)
        if callable(f): f()
        self.assert_gradient_firewall()

    def save(self,path:str|Path)->None:
        f=getattr(self.engine,"save",None)
        if not callable(f): raise UnsupportedOperation("external E0 engine has no save()")
        f(path)

    def restore(self,path:str|Path)->None:
        f=getattr(self.engine,"restore",None) or getattr(self.engine,"load",None)
        if not callable(f): raise UnsupportedOperation("external E0 engine has no restore/load")
        f(path); self.assert_gradient_firewall()

    def budget_report(self)->BudgetReport:
        f=getattr(self.engine,"budget_report",None)
        if callable(f):
            r=f();
            if isinstance(r,BudgetReport): return r
            if isinstance(r,dict): return BudgetReport(**r)
        active=getattr(self.engine,"active_synapse_count",None)
        n=int(active() if callable(active) else active) if active is not None else 0
        return BudgetReport(n, n or None, "external E0 reported active synapses; 0 means unavailable")

    def compute_report(self)->ComputeReport:
        return ComputeReport(self._inference_steps,self._plasticity_steps,0,0)

    def lesion(self,unit_indices:list[int])->None:
        f=getattr(self.engine,"lesion",None)
        if not callable(f): raise UnsupportedOperation("external E0 engine has no lesion()")
        f(unit_indices); self.assert_gradient_firewall()

def _accepts_dt(f)->bool:
    import inspect
    try: return len(inspect.signature(f).parameters)>=2
    except Exception: return True
