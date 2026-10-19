from __future__ import annotations
from collections import deque
import torch
from .contracts import SubstrateState

class DenseStateFeatures:
    def __call__(self,state:SubstrateState)->torch.Tensor: return state.vector.float().flatten()

class SpikeFeatures:
    def __init__(self, tau_ms:float=25.0): self.tau_ms=float(tau_ms); self._trace=None
    def reset(self): self._trace=None
    def __call__(self,state:SubstrateState,dt:float=0.01)->torch.Tensor:
        x=state.vector.float().flatten(); decay=float(torch.exp(torch.tensor(-dt/(self.tau_ms/1000.0))))
        self._trace=x.clone() if self._trace is None else self._trace*decay+x
        return self._trace.clone()

class RawHistoryFeatures:
    def __init__(self,lags:int=20): self.lags=int(lags); self._q=deque(maxlen=self.lags)
    def reset(self): self._q.clear()
    def __call__(self,x:torch.Tensor)->torch.Tensor:
        x=torch.as_tensor(x,dtype=torch.float32).flatten(); self._q.append(x)
        missing=self.lags-len(self._q)
        return torch.cat([torch.zeros(missing*x.numel()), *self._q])
