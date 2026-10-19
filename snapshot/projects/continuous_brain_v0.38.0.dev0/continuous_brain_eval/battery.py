from __future__ import annotations
from pathlib import Path
import tempfile,torch
from continuous_brain_substrates.contracts import Substrate,TrainingBatch,UnsupportedOperation,GradientFirewallError
from .status import EvidenceStatus,EngineeringResult
from .synthetic import pulse_stream,rhythmic_stream

class RequiresE2Body(RuntimeError): pass
class EngineeringBattery:
    """API/provenance smoke harness. These are not preregistered H1/H2 experiments."""
    def __init__(self,input_dim:int): self.input_dim=input_dim
    def run(self,substrate:Substrate)->list[EngineeringResult]:
        return [self.b1(substrate),self.b2(substrate),self.b3(substrate),self.b4(substrate),self.b5(substrate),self.b6(substrate)]
    def _r(self,item,diag=None,notes=None,status=EvidenceStatus.ENGINEERING_ONLY): return EngineeringResult(item,status,diag or {},notes or [])
    def b1(self,substrate): return self._r('B1',status=EvidenceStatus.REQUIRES_E2_BODY,notes=['real E2 MuJoCo body is required; no stand-in is used'])
    def b2(self,substrate):
        seq=pulse_stream(24,self.input_dim,17); substrate.reset(17); diag={}
        try:
            d=substrate.train_step(TrainingBatch(seq)); diag.update({f'train_{k}':v for k,v in d.items()}); diag['learning_api']='gradient'
        except GradientFirewallError:
            try: substrate.plasticity_step({'engineering_signal':0.0}); diag['learning_api']='local_plasticity'
            except UnsupportedOperation as e: diag['learning_api']='unavailable'; diag['detail']=str(e)
        except UnsupportedOperation as e: diag['learning_api']='unavailable'; diag['detail']=str(e)
        diag['compute']=substrate.compute_report().__dict__
        return self._r('B2',diag,['synthetic API exercise only; no continual-learning metric is preregistered'],EvidenceStatus.UNREGISTERED_METRIC)
    def b3(self,substrate):
        x=pulse_stream(8,self.input_dim,2); substrate.reset(2)
        for row in x[:4]: substrate.step(row,0.01)
        before=substrate.expose_state().vector.clone()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.pt'
            try:
                substrate.save(p); substrate.step(x[4],0.01); substrate.restore(p); after=substrate.expose_state().vector
                exact=bool(torch.equal(before,after)); diag={'save_restore_supported':True,'state_exact':exact}
            except UnsupportedOperation as e: diag={'save_restore_supported':False,'detail':str(e)}
        return self._r('B3',diag,['engineering save/restore check only'],EvidenceStatus.UNREGISTERED_METRIC)
    def b4(self,substrate):
        a=pulse_stream(16,self.input_dim,4); substrate.reset(4); substrate.step(torch.ones(self.input_dim),0.01)
        for row in a: substrate.step(row,0.01)
        s1=substrate.expose_state().vector.float(); substrate.reset(4); substrate.step(torch.zeros(self.input_dim),0.01)
        for row in a: substrate.step(row,0.01)
        s0=substrate.expose_state().vector.float(); d=float(torch.linalg.vector_norm(s1-s0)) if s1.numel()==s0.numel() else float('nan')
        return self._r('B4',{'delayed_state_difference_norm':d},['state sensitivity is not a working-memory admission'],EvidenceStatus.UNREGISTERED_METRIC)
    def b5(self,substrate):
        seq=rhythmic_stream(30,self.input_dim); substrate.reset(5); states=[]
        for row in seq: states.append(substrate.step(row,0.01).vector.float())
        stack=torch.stack(states); temporal_var=float(stack.var(dim=0,unbiased=False).mean())
        return self._r('B5',{'mean_state_temporal_variance':temporal_var},['not a preregistered temporal-structure metric'],EvidenceStatus.UNREGISTERED_METRIC)
    def b6(self,substrate):
        substrate.reset(6); substrate.step(torch.ones(self.input_dim),0.01); n=substrate.expose_state().vector.numel(); idx=list(range(max(1,n//10)))
        try: substrate.lesion(idx); out=substrate.step(torch.ones(self.input_dim),0.01); diag={'lesion_supported':True,'lesioned_units':len(idx),'post_state_norm':float(out.vector.float().norm())}
        except UnsupportedOperation as e: diag={'lesion_supported':False,'detail':str(e)}
        return self._r('B6',diag,['lesion API exercise only; no recovery margin is preregistered'],EvidenceStatus.UNREGISTERED_METRIC)
