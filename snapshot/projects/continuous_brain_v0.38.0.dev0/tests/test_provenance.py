import json
from continuous_brain_substrates.provenance import RunRecord,RunRegistry

def test_content_addressed_record(tmp_path):
    r=RunRecord('r1','ENGINEERING_ONLY','dense',1,'abc'); p=RunRegistry(tmp_path).write(r); d=json.loads(p.read_text()); assert d['record_sha256']==r.digest(); assert d['scientific_interpretation_permitted'] is False
