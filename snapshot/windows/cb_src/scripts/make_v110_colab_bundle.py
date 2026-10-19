#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, tarfile
from pathlib import Path


def sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description='Bundle a v0.11.0 expanded child and selected nurseries for Colab.')
    p.add_argument('--project-root',default='.')
    p.add_argument('--state-dir',required=True)
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    project=Path(args.project_root).resolve(); state=Path(args.state_dir).resolve(); datasets=[Path(x).resolve() for x in args.datasets]
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    if not (project/'pyproject.toml').exists(): raise SystemExit('project-root does not look like Continuous Brain')
    if not (state/'infant_training.pt').exists(): raise SystemExit('state-dir missing infant_training.pt')
    for d in datasets:
        if not (d/'lessons.jsonl').exists(): raise SystemExit(f'dataset missing lessons.jsonl: {d}')
    manifest={
      'format':'continuous_brain.colab_bundle.v110',
      'state_checkpoint_sha256':sha(state/'infant_training.pt'),
      'state_basename':state.name,
      'datasets':[d.name for d in datasets],
    }
    temp=out.with_suffix(out.suffix+'.tmp')
    with tarfile.open(temp,'w:gz') as tf:
        def filt(info):
            name=info.name
            if any(part in name for part in ('/.git/','/__pycache__/','/.pytest_cache/')): return None
            if name.endswith(('.pyc','.pyo')): return None
            return info
        tf.add(project,arcname='continuous_brain_project',filter=filt)
        tf.add(state,arcname=f'payload/state/{state.name}',filter=filt)
        seen=set()
        for d in datasets:
            arc=f'payload/datasets/{d.name}'
            if arc in seen: raise SystemExit(f'duplicate dataset basename: {d.name}')
            seen.add(arc); tf.add(d,arcname=arc,filter=filt)
        data=json.dumps(manifest,indent=2,sort_keys=True).encode()
        import io
        ti=tarfile.TarInfo('payload/BUNDLE_MANIFEST.json'); ti.size=len(data)
        tf.addfile(ti,io.BytesIO(data))
    temp.replace(out)
    print(json.dumps({**manifest,'output':str(out),'bundle_sha256':sha(out)},indent=2,sort_keys=True))

if __name__=='__main__': main()
