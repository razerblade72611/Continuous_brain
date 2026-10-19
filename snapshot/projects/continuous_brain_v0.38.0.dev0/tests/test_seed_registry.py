import json,pytest
from continuous_brain_substrates.seed_registry import SeedRegistry,SeedRange

def test_overlap_rejected(tmp_path):
    p=tmp_path/'s.json'; p.write_text(json.dumps({'ranges':[{'start':10,'end':20,'purpose':'spent','status':'SPENT','source':'x'}]})); r=SeedRegistry(p)
    with pytest.raises(ValueError): r.reserve(SeedRange(20,30,'bad'))
    r.reserve(SeedRange(21,30,'ok')); assert len(r.load())==2
