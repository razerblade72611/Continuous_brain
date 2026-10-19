import torch
from continuous_brain_substrates import DenseRNNConfig,DenseRNNSubstrate,TrainingBatch,UnsupportedOperation

def test_dense_rnn_train_save_restore(tmp_path):
    s=DenseRNNSubstrate(DenseRNNConfig(input_dim=4,hidden_dim=7,seed=3)); seq=torch.randn(12,4); d=s.train_step(TrainingBatch(seq)); assert d['prediction_mse']>=0
    s.reset(9); s.step(torch.ones(4),.01); before=s.expose_state().vector.clone(); p=tmp_path/'x.pt'; s.save(p); s.step(torch.zeros(4),.01); s.restore(p); assert torch.equal(before,s.expose_state().vector)
    assert s.budget_report().learned_scalars>0

def test_dense_local_plasticity_is_not_aliased_to_gradient():
    s=DenseRNNSubstrate(DenseRNNConfig(input_dim=3,hidden_dim=5))
    try: s.plasticity_step({})
    except UnsupportedOperation: pass
    else: raise AssertionError('must not alias local plasticity to gradient training')
