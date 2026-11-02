"""Execute only the extracted phase-2 JOB locally, with model/GPU calls stubbed.
No boot, RunPod, SSH, keys, real model or GPU access is performed.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
CODE='38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae'
MODEL='69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b'
GPU='NVIDIA RTX 6000 Ada Generation'
STUB=r'''
import json,os,re,sys
from pathlib import Path
a=sys.argv[1:]
if a[0]=='-c':os.execv(sys.executable,[sys.executable,*a])
with open(os.environ['MOCK_LOG'],'a') as f:f.write(json.dumps(a)+'\n')
def opt(k,default=None):return a[a.index(k)+1] if k in a else default
if a[:2]==['scripts/support1.py','codehash']:print(os.environ['MOCK_CODE']);sys.exit(0)
if a[:2]==['scripts/support1.py','train']:
 pf=Path(opt('--prep'));assert pf.is_file();s=int(re.search(r'prep_seed(\d+)\.json',pf.name).group(1));d=Path(opt('--out'));d.mkdir(parents=True,exist_ok=True)
 assert '--fork' in a and opt('--model-manifest-sha')==os.environ['MOCK_MODEL']
 assert opt('--validator-report')=='results/support1/phase0/validation.json'
 assert opt('--calibration-result')=='results/support1/phase1/cal/calcheck.json'
 assert opt('--calibration-prep')=='results/support1/phase0/prep_seed24.json'
 (d/f'train_seed{s}.json').write_text(json.dumps({'mock':True,'seed':s}))
 for arm in ['P','U1','G1','U4','G4']:
  for L in range(1,6):(d/f'cortex_{arm}_s{L}.pt').write_bytes(b'mock checkpoint')
 for arm in ['U4','G4']:(d/f'cortex_{arm}_s4_withheld.pt').write_bytes(b'mock fork checkpoint')
 sys.exit(0)
if a[:2]==['scripts/support1.py','evalck']:
 pf=Path(opt('--prep'));assert pf.is_file();seed=int(re.search(r'prep_seed(\d+)\.json',pf.name).group(1));arm=opt('--arm','N');L=int(opt('--sleep'));mode=opt('--mode','base');h=opt('--history','observed')
 assert opt('--qdiag')=='64' and opt('--model-manifest-sha')==os.environ['MOCK_MODEL']
 if arm!='N':assert Path(opt('--ckpt')).is_file() and mode in ['merged','unmerged']
 if h=='withheld':assert arm in ['U4','G4'] and L==4
 out=Path(opt('--out'));suffix='' if arm=='N' else '_'+mode
 expected=f'eval_{arm}_s{L}'+('_withheld' if h=='withheld' else '')+suffix+'.json'
 assert out.name==expected
 out.write_text(json.dumps({'mock':True,'seed':seed,'arm':arm,'sleep':L,'mode':mode,'history':h}))
 sys.exit(0)
if a[0] in ['scripts/support1_analyze.py','scripts/support1_sol_analyze.py']:
 if 'sol_analyze' in a[0]:
  assert opt('--phase0-dir')=='results/support1/phase0' and opt('--phase1-dir')=='results/support1/phase1'
  assert '--fork' in a and opt('--gpu')==os.environ['MOCK_GPU']
 Path(opt('--out')).write_text(json.dumps({'verdict':'MOCK_COMPLETE'}))
 sys.exit(int(os.environ['MOCK_SOL_RC']) if 'sol_analyze' in a[0] else 0)
raise AssertionError(a)
'''


@pytest.mark.parametrize('case',['normal','sol_failure','wrong_gpu'])
def test_phase2_job_matrix_and_preservation(tmp_path,case):
 ws=tmp_path/'workspace';repo=ws/'Continuous_brain_support1';(repo/'results/support1').mkdir(parents=True)
 for name in ['phase0','phase1']:(repo/'results/support1'/name).symlink_to(ROOT/f'results/support1/{name}',target_is_directory=True)
 (ws/'model_manifest.sha256').write_text(MODEL)
 interpreter=ws/'venv/bin/python';interpreter.parent.mkdir(parents=True)
 interpreter.write_text('#!'+sys.executable+'\n'+STUB);interpreter.chmod(0o755)
 binpath=ws/'bin';binpath.mkdir();gpu=binpath/'nvidia-smi';gpu.write_text('#!/bin/sh\nprintf "%s\\n" "$MOCK_GPU"\n');gpu.chmod(0o755)
 source=(ROOT/'scripts/cb_support1_p2.sh').read_text();parts=re.findall(r"<<'JOB'[^\n]*\n(.*?)\nJOB\n",source,re.S);assert len(parts)==1
 script=ws/'job.sh';script.write_text(parts[0].replace('/workspace',str(ws)))
 log=ws/'events.jsonl';log.touch()
 env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],MOCK_LOG=str(log),MOCK_CODE=CODE,MOCK_MODEL=MODEL,
          MOCK_GPU='' if case=='wrong_gpu' else GPU,MOCK_SOL_RC='2' if case=='sol_failure' else '0')
 proc=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True,timeout=180)
 events=[json.loads(s) for s in log.read_text().splitlines()];r=ws/'support1'
 assert (ws/'support1_phase2.tar.gz').is_file(),proc.stdout+proc.stderr
 if case=='wrong_gpu':
  assert proc.returncode==1 and not any(e[:2]==['scripts/support1.py','train'] for e in events)
 else:
  assert proc.returncode==0,proc.stdout+proc.stderr
  assert sum(e[:2]==['scripts/support1.py','train'] for e in events)==4
  assert sum(e[:2]==['scripts/support1.py','evalck'] for e in events)==236
  assert len(list(r.rglob('*.pt')))==108 and len(list(r.glob('seed*/eval_*.json')))==236
  assert sum(e[0]=='scripts/support1_analyze.py' for e in events)==1
  assert sum(e[0]=='scripts/support1_sol_analyze.py' for e in events)==1
  assert 'SOL ANALYZER rc='+env['MOCK_SOL_RC'] in proc.stdout
  with tarfile.open(ws/'support1_phase2.tar.gz') as t:
   names=t.getnames();assert sum('/eval_' in n and n.endswith('.json') for n in names)==236
   assert not any(n.endswith('.pt') for n in names)
   assert any(n.endswith('analysis_sol.json') for n in names)
