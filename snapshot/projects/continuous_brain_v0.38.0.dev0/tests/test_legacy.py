import hashlib,json
from pathlib import Path
from continuous_brain_unified.legacy import LegacyManifest,LegacyVerifier,BindingState

def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_hash_pinned_binding_and_mismatch(tmp_path):
    pkg=tmp_path/'continuous_brain'; pkg.mkdir(); (pkg/'__init__.py').write_text(''); mod=pkg/'x.py'; mod.write_text('class X: pass\n'); ev=tmp_path/'eval_x.py'; ev.write_text('x=1\n')
    m={'schema_version':1,'source_snapshot':{},'components':[{'component_id':'x','scientific_status':'H','module':'continuous_brain.x','source_files':[{'path':'continuous_brain/x.py','sha256':h(mod),'role':'source'}],'evaluator_files':[{'path':'eval_x.py','sha256':h(ev),'role':'evaluator'}],'entry_symbols':['X'],'locally_bindable':True}]}
    mp=tmp_path/'m.json'; mp.write_text(json.dumps(m)); man=LegacyManifest.load(mp); v=LegacyVerifier(man,tmp_path); r=v.verify_component(man.get('x'),bind=True); assert r.state==BindingState.ENTRYPOINT_BOUND
    mod.write_text('class X: pass\n# changed\n'); r=v.verify_component(man.get('x')); assert r.state==BindingState.UNBOUND; assert 'hash mismatch' in r.reasons[0]
