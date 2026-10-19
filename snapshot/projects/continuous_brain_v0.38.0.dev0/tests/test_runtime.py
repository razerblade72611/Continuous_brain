import json
from continuous_brain_unified.runtime import UnifiedRuntime

def test_runtime_reports_unbound_without_standins(tmp_path):
    m={'schema_version':1,'source_snapshot':{},'components':[{'component_id':'missing','scientific_status':'H','module':'continuous_brain.nope','source_files':[{'path':'continuous_brain/nope.py','sha256':'0'*64,'role':'source'}],'evaluator_files':[],'entry_symbols':['Nope'],'locally_bindable':True}]}
    p=tmp_path/'m.json'; p.write_text(json.dumps(m)); r=UnifiedRuntime(p,tmp_path); s=r.boot(); assert s.components['missing']=='UNBOUND'; assert not s.scientific_integration_claimed
