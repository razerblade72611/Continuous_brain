#!/usr/bin/env python3
from pathlib import Path
import json,os
from continuous_brain_unified.runtime import UnifiedRuntime
ROOT=Path(__file__).resolve().parents[1]
r=UnifiedRuntime(ROOT/'registry'/'legacy_manifest.json',os.environ.get('CONTINUOUS_BRAIN_LEGACY_ROOT')); print(json.dumps(r.boot().__dict__,indent=2,sort_keys=True))
