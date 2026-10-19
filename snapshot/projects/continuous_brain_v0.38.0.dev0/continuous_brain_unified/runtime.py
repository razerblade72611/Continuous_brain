from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from .legacy import BindingReport,BindingState,LegacyManifest,LegacyVerifier
from .coprocessor import Coprocessor,NullCoprocessor

@dataclass
class RuntimeStatus:
    version:str
    components:dict[str,str]
    unbound:dict[str,list[str]]
    scientific_integration_claimed:bool=False
    coprocessor_enabled:bool=False

class UnifiedRuntime:
    """Composition shell. It does not reimplement historical cognition."""
    def __init__(self,manifest_path:str|Path,legacy_root:str|Path|None=None,coprocessor:Coprocessor|None=None):
        self.manifest=LegacyManifest.load(manifest_path); self.verifier=LegacyVerifier(self.manifest,legacy_root)
        self.coprocessor=coprocessor or NullCoprocessor(); self.bindings:dict[str,BindingReport]={}
    def boot(self,bind_entrypoints:bool=True)->RuntimeStatus:
        self.bindings={r.component_id:r for r in self.verifier.audit(bind=bind_entrypoints)}
        return self.status()
    def status(self)->RuntimeStatus:
        comps={k:v.state.value for k,v in self.bindings.items()}; unbound={k:v.reasons for k,v in self.bindings.items() if v.state==BindingState.UNBOUND}
        return RuntimeStatus("0.38.0.dev0",comps,unbound,False,not isinstance(self.coprocessor,NullCoprocessor))
    def component(self,component_id:str)->BindingReport:
        if component_id not in self.bindings: raise KeyError(f'component not booted: {component_id}')
        r=self.bindings[component_id]
        if r.state!=BindingState.ENTRYPOINT_BOUND: raise RuntimeError(f'{component_id} is {r.state.value}: {r.reasons}')
        return r
    def symbol(self,component_id:str,symbol:str):
        r=self.component(component_id)
        if symbol not in r.symbols: raise KeyError(f'{symbol} is not a declared verified entrypoint for {component_id}')
        return r.symbols[symbol]
