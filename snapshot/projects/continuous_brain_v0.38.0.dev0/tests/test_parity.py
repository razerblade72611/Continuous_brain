from continuous_brain_eval.parity import power_aware_rate_parity

def test_small_population_can_be_marked_underpowered():
    r=power_aware_rate_parity(2.0,2.1,10,60); assert r.tolerance>0; assert isinstance(r.underpowered,bool)
