#!/usr/bin/env python3
import argparse,json
from dataclasses import asdict
from continuous_brain_substrates.dense_rnn import DenseRNNConfig,DenseRNNSubstrate
from continuous_brain_eval.battery import EngineeringBattery
p=argparse.ArgumentParser(); p.add_argument('--substrate',choices=['dense'],default='dense'); p.add_argument('--input-dim',type=int,default=8); p.add_argument('--hidden-dim',type=int,default=32); a=p.parse_args()
s=DenseRNNSubstrate(DenseRNNConfig(a.input_dim,a.hidden_dim)); results=EngineeringBattery(a.input_dim).run(s); print(json.dumps([asdict(x) for x in results],indent=2,default=str))
