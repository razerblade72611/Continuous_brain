import torch,pytest
from continuous_brain_substrates.snn_adapter import ExternalSNNAdapter
from continuous_brain_substrates.contracts import GradientFirewallError,TrainingBatch
class FakeE0:
    def __init__(self): self.state=torch.zeros(3); self.w=torch.zeros(1,requires_grad=False)
    def reset(self,seed=None): self.state.zero_()
    def step(self,x,dt): self.state=self.state+torch.as_tensor(x)[:3]; return self.state
    def plasticity_step(self,m): self.state += 0
    def lesion(self,idx): self.state[idx]=0

def test_snn_train_step_always_rejected():
    a=ExternalSNNAdapter(FakeE0()); a.step(torch.ones(3),.001); a.plasticity_step({})
    with pytest.raises(GradientFirewallError): a.train_step(TrainingBatch(torch.zeros(2,3)))

def test_snn_requires_grad_detected():
    f=FakeE0(); f.w.requires_grad_(True)
    with pytest.raises(GradientFirewallError): ExternalSNNAdapter(f)
