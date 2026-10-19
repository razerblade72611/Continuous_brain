from continuous_brain_eval.indicators import INDICATORS,empty_evidence_registry

def test_fourteen_indicators_and_no_admissions():
    assert len(INDICATORS)==14; r=empty_evidence_registry(); assert all(x['v038_status']=='NOT_ASSESSED' and not x['scientific_interpretation_permitted'] for x in r)
