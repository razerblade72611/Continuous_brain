from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import hashlib, importlib, json, os, sys

class BindingState(str,Enum):
    UNBOUND="UNBOUND"
    SOURCE_VERIFIED="SOURCE_VERIFIED"
    ENTRYPOINT_BOUND="ENTRYPOINT_BOUND"
    SCIENTIFICALLY_INTEGRATED="SCIENTIFICALLY_INTEGRATED"  # never emitted by this release

@dataclass(frozen=True)
class FileSpec:
    path:str; sha256:str; role:str="source"

@dataclass(frozen=True)
class ComponentSpec:
    component_id:str
    scientific_status:str
    module:str|None
    source_files:tuple[FileSpec,...]=()
    evaluator_files:tuple[FileSpec,...]=()
    entry_symbols:tuple[str,...]=()
    notes:str=""
    locally_bindable:bool=True

@dataclass
class LegacyManifest:
    schema_version:int
    source_snapshot:dict[str,Any]
    components:list[ComponentSpec]
    @classmethod
    def load(cls,path:str|Path)->'LegacyManifest':
        d=json.loads(Path(path).read_text())
        comps=[]
        for c in d['components']:
            sf=tuple(FileSpec(**x) for x in c.get('source_files',[])); ef=tuple(FileSpec(**x) for x in c.get('evaluator_files',[]))
            comps.append(ComponentSpec(c['component_id'],c['scientific_status'],c.get('module'),sf,ef,tuple(c.get('entry_symbols',[])),c.get('notes',''),c.get('locally_bindable',True)))
        return cls(d['schema_version'],d.get('source_snapshot',{}),comps)
    def get(self,cid:str)->ComponentSpec:
        return next(c for c in self.components if c.component_id==cid)

@dataclass
class BindingReport:
    component_id:str
    state:BindingState
    reasons:list[str]=field(default_factory=list)
    symbols:dict[str,Any]=field(default_factory=dict,repr=False)
    module:Any=None

class LegacyVerifier:
    def __init__(self,manifest:LegacyManifest,root:str|Path|None=None):
        self.manifest=manifest; self.root=Path(root or os.environ.get('CONTINUOUS_BRAIN_LEGACY_ROOT','')).expanduser() if (root or os.environ.get('CONTINUOUS_BRAIN_LEGACY_ROOT')) else None
    @staticmethod
    def _sha(path:Path)->str:
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
        return h.hexdigest()
    def _resolve(self,rel:str)->Path:
        if self.root is None: raise FileNotFoundError('CONTINUOUS_BRAIN_LEGACY_ROOT is not set')
        p=(self.root/rel).resolve(); rr=self.root.resolve()
        if rr not in p.parents and p!=rr: raise ValueError(f'path escapes legacy root: {rel}')
        return p
    def verify_component(self,spec:ComponentSpec,bind:bool=False)->BindingReport:
        reasons=[]
        if not spec.locally_bindable: return BindingReport(spec.component_id,BindingState.UNBOUND,[spec.notes or 'manifest marks component unbound'])
        if self.root is None: return BindingReport(spec.component_id,BindingState.UNBOUND,['legacy root not configured'])
        for fs in (*spec.source_files,*spec.evaluator_files):
            p=self._resolve(fs.path)
            if not p.is_file(): reasons.append(f'missing {fs.role}: {fs.path}'); continue
            got=self._sha(p)
            if got!=fs.sha256: reasons.append(f'hash mismatch {fs.path}: expected {fs.sha256}, got {got}')
        if reasons: return BindingReport(spec.component_id,BindingState.UNBOUND,reasons)
        if not bind or not spec.module: return BindingReport(spec.component_id,BindingState.SOURCE_VERIFIED,[])
        try: module=self._import_exact(spec.module)
        except Exception as e: return BindingReport(spec.component_id,BindingState.UNBOUND,[f'import failed: {type(e).__name__}: {e}'])
        symbols={}; missing=[]
        for s in spec.entry_symbols:
            if hasattr(module,s): symbols[s]=getattr(module,s)
            else: missing.append(s)
        if missing: return BindingReport(spec.component_id,BindingState.UNBOUND,[f'missing symbol(s): {missing}'],module=module)
        return BindingReport(spec.component_id,BindingState.ENTRYPOINT_BOUND,[],symbols,module)
    def _import_exact(self,module_name:str):
        if self.root is None: raise RuntimeError('legacy root not configured')
        pkg=sys.modules.get('continuous_brain')
        if pkg is not None:
            f=Path(getattr(pkg,'__file__','')).resolve()
            if self.root.resolve() not in f.parents:
                raise RuntimeError(f'continuous_brain already imported from different root: {f}')
        sys.path.insert(0,str(self.root.resolve()))
        try:
            m=importlib.import_module(module_name)
            mf=Path(m.__file__).resolve()
            if self.root.resolve() not in mf.parents: raise RuntimeError(f'module resolved outside verified root: {mf}')
            return m
        finally:
            try: sys.path.remove(str(self.root.resolve()))
            except ValueError: pass
    def audit(self,bind:bool=False)->list[BindingReport]: return [self.verify_component(c,bind=bind) for c in self.manifest.components]
