from __future__ import annotations
import argparse,json,os
from pathlib import Path
from importlib.resources import files
from .runtime import UnifiedRuntime

def main(argv=None):
    default_manifest=str(files('continuous_brain_unified').joinpath('data/legacy_manifest.json')); p=argparse.ArgumentParser(); p.add_argument('--manifest',default=default_manifest); p.add_argument('--legacy-root',default=os.environ.get('CONTINUOUS_BRAIN_LEGACY_ROOT')); p.add_argument('--no-bind',action='store_true')
    a=p.parse_args(argv); r=UnifiedRuntime(a.manifest,a.legacy_root); s=r.boot(bind_entrypoints=not a.no_bind); print(json.dumps(s.__dict__,indent=2,sort_keys=True))
if __name__=='__main__': main()
