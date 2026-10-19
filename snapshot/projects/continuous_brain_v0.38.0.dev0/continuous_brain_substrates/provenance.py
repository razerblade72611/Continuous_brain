from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib, json, os, subprocess, tempfile

def canonical_json(obj:Any)->bytes:
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha256_file(path:str|Path)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def object_hash(obj:Any)->str: return sha256_bytes(canonical_json(obj))

def git_revision(root:str|Path='.') -> str:
    try: return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "UNAVAILABLE"

@dataclass
class ArtifactRef:
    path:str; sha256:str
    @classmethod
    def from_path(cls,path:str|Path): return cls(str(path),sha256_file(path))

@dataclass
class RunRecord:
    run_id:str
    status:str
    substrate:str
    seed:int|None
    config_hash:str
    code_revision:str="UNAVAILABLE"
    scientific_interpretation_permitted:bool=False
    started_at_utc:str=field(default_factory=lambda:datetime.now(timezone.utc).isoformat())
    compute:dict[str,Any]=field(default_factory=dict)
    artifacts:list[dict[str,str]]=field(default_factory=list)
    notes:list[str]=field(default_factory=list)
    def digest(self)->str: return object_hash(asdict(self))

class RunRegistry:
    def __init__(self,root:str|Path): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def write(self,record:RunRecord)->Path:
        d=asdict(record); d["record_sha256"]=record.digest()
        run_dir=self.root/record.run_id; run_dir.mkdir(parents=True,exist_ok=True)
        p=run_dir/'manifest.json'; _atomic_json(p,d)
        with open(self.root/'run_registry.jsonl','a',encoding='utf-8') as f: f.write(json.dumps(d,sort_keys=True)+'\n')
        return p

def _atomic_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent); os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
