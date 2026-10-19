from continuous_brain_substrates import DenseRNNConfig,DenseRNNSubstrate
from continuous_brain_eval import EngineeringBattery,EvidenceStatus

def test_battery_never_claims_scientific_pass():
    s=DenseRNNSubstrate(DenseRNNConfig(input_dim=4,hidden_dim=8)); rs=EngineeringBattery(4).run(s)
    assert len(rs)==6; assert rs[0].status==EvidenceStatus.REQUIRES_E2_BODY
    assert all(not r.scientific_interpretation_permitted for r in rs)
    assert all(r.status in {EvidenceStatus.REQUIRES_E2_BODY,EvidenceStatus.ENGINEERING_ONLY,EvidenceStatus.UNREGISTERED_METRIC} for r in rs)
