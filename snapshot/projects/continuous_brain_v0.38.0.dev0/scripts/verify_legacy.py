#!/usr/bin/env python3
from pathlib import Path
import json,os
from continuous_brain_unified.legacy import LegacyManifest,LegacyVerifier
ROOT=Path(__file__).resolve().parents[1]
m=LegacyManifest.load(ROOT/'registry'/'legacy_manifest.json'); v=LegacyVerifier(m,os.environ.get('CONTINUOUS_BRAIN_LEGACY_ROOT'))
rows=[]
for r in v.audit(bind=False): rows.append({'component':r.component_id,'state':r.state.value,'reasons':r.reasons})
print(json.dumps(rows,indent=2))
raise SystemExit(1 if any(x['state']=='UNBOUND' and not any(c.component_id==x['component'] and not c.locally_bindable for c in m.components) for x in rows) else 0)
