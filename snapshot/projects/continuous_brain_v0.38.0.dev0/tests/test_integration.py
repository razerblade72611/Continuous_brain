from continuous_brain_eval.integration import IntegrationPlan,validate_prediction_matrix
from continuous_brain_unified.interventions import InterventionSpec,PredictionCell,ExpectedEffect

def test_prediction_matrix_requires_every_cell_and_rationale():
    p=IntegrationPlan([InterventionSpec('erase','auto','erase')],[PredictionCell('erase','auto',ExpectedEffect.CHANGE,'causal test')])
    assert validate_prediction_matrix(p,['auto'])==[]
    assert validate_prediction_matrix(p,['auto','plan'])
