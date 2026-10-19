from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
from .runtime import UnifiedRuntime
from continuous_brain_substrates.provenance import object_hash

def save_shell_checkpoint(runtime:UnifiedRuntime,path:str|Path)->Path:
    """Save composition metadata only; never pretends to serialize legacy mechanism state."""
    status=asdict(runtime.status()); d={"kind":"v038_shell_checkpoint","status":status,"status_sha256":object_hash(status)}
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); return p

def verify_shell_checkpoint(path:str|Path)->dict:
    d=json.loads(Path(path).read_text()); expected=d['status_sha256']; got=object_hash(d['status'])
    if expected!=got: raise ValueError('shell checkpoint hash mismatch')
    return d
