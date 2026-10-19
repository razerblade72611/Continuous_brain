from __future__ import annotations
import torch

def pulse_stream(length:int,input_dim:int,seed:int=1)->torch.Tensor:
    g=torch.Generator().manual_seed(seed); x=0.02*torch.randn(length,input_dim,generator=g)
    for t in range(5,length,17): x[t:min(t+3,length), t%input_dim]+=1.0
    return x

def rhythmic_stream(length:int,input_dim:int)->torch.Tensor:
    t=torch.arange(length,dtype=torch.float32); cols=[]
    for i in range(input_dim): cols.append(torch.sin(t*(0.05+0.01*i))+0.25*torch.sin(t*(0.13+0.02*i)))
    return torch.stack(cols,dim=1)
