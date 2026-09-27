"""G0 isolated legacy requalification runner."""
from __future__ import annotations
import json, os, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from continuous_brain_protocols import ResultStatus, SeedPurpose, sha256_file, sha256_json
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_unified.prereg import PreregistrationRegistry, PreregistrationError
from .g0 import G0Plan

RECIPE_FORMAT='cb-g0-recipes/1'
SEED_FORMAT='cb-g0-seed-plan/1'

@dataclass(frozen=True)
class G0Recipe:
    mechanism_id: str; evaluator: str; seed_symbol: str; seed_container: str; pass_field: str; fresh_family: str

@dataclass(frozen=True)
class G0Prepared:
    mechanism_id: str; evidence_ok: bool; prereg_status: str; recipe: G0Recipe|None; seed_plan: dict|None; blocked_reasons: tuple[str,...]

def _load_json(path): return json.loads(Path(path).read_text())

def load_recipes(path='legacy/g0_recipes.json'):
    raw=_load_json(path)
    if raw.get('format')!=RECIPE_FORMAT: raise ValueError('bad G0 recipe format')
    return ({r['mechanism_id']:G0Recipe(**r) for r in raw['recipes']},{r['mechanism_id']:r['reason'] for r in raw.get('blocked',[])})

def load_seed_plan(path='legacy/g0_seed_plan.json'):
    raw=_load_json(path)
    if raw.get('format')!=SEED_FORMAT: raise ValueError('bad G0 seed-plan format')
    return {f['name']:f for f in raw['families']}

def _tree_metadata(root: Path)->str:
    rows=[]
    for p in sorted(root.rglob('*')):
        if p.is_file():
            st=p.stat(); rows.append((str(p.relative_to(root)),st.st_size,st.st_mtime_ns))
    return sha256_json(rows)

def _nested(obj:dict,path:str):
    cur=obj
    for part in path.split('.'):
        if not isinstance(cur,dict) or part not in cur: raise KeyError(path)
        cur=cur[part]
    return cur

def prepare(mechanism_id:str, legacy_root:str|Path, plan_path='legacy/g0_plan.json', registry_dir=None)->G0Prepared:
    plan=G0Plan.load(plan_path); v=plan.verify({'wsl-canonical':Path(legacy_root).expanduser().resolve()},mechanism_id)[0]
    recipes,blocked=load_recipes(); seeds=load_seed_plan(); recipe=recipes.get(mechanism_id); reasons=[]
    if not v.ok: reasons.append('pinned G0 evidence hashes do not match')
    if mechanism_id in blocked: reasons.append(blocked[mechanism_id])
    seed=None if recipe is None else seeds.get(recipe.fresh_family)
    if recipe is not None and seed is None: reasons.append('recipe seed family missing from proposal')
    preg=PreregistrationRegistry.load(registry_dir).entries.get('G0-LEGACY-REQUAL-v1')
    return G0Prepared(mechanism_id,v.ok,'MISSING' if preg is None else preg.status,recipe,seed,tuple(reasons))

def _bootstrap_source(evaluator:Path, seed_symbol:str, start:int, stop:int, container:str, state_dir:Path)->str:
    lines=[
      'import importlib.util, pathlib, sys',
      'sys.dont_write_bytecode=True',
      f'root=pathlib.Path({str(evaluator.parent)!r}).resolve()',
      'sys.path.insert(0,str(root))',
      f'spec=importlib.util.spec_from_file_location("cb_g0_frozen_evaluator", {str(evaluator)!r})',
      'm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)',
      f'old=getattr(m,{seed_symbol!r})',
      f'vals=range({start},{stop+1})',
      f'setattr(m,{seed_symbol!r}, list(vals) if {container!r}=="list" else range({start},{stop+1}))',
      f'sys.argv=[{str(evaluator)!r},"--state-dir",{str(state_dir)!r}]',
      'm.main()',
    ]
    return '\n'.join(lines)+'\n'

def run(mechanism_id:str, legacy_root:str|Path, output_root:str|Path, *, registry_dir=None, plan_path='legacy/g0_plan.json')->dict[str,Any]:
    prepared=prepare(mechanism_id,legacy_root,plan_path,registry_dir)
    if prepared.blocked_reasons:
        return {'item':'G0','mechanism_id':mechanism_id,'status':ResultStatus.BLOCKED.value,'reasons':list(prepared.blocked_reasons),'prereg_status':prepared.prereg_status}
    try:
        preg=PreregistrationRegistry.load(registry_dir).require('G0-LEGACY-REQUAL-v1',item='G0')
    except PreregistrationError as e:
        raise PreregistrationError('G0 execution is locked until David approves G0-LEGACY-REQUAL-v1: '+str(e)) from e
    recipe=prepared.recipe; seed=prepared.seed_plan
    assert recipe and seed
    reg=SeedRegistry.load(registry_dir); fam=reg.get(seed['name'])
    if fam.purpose!=SeedPurpose.DEV or fam.start!=seed['start'] or fam.stop!=seed['stop']: raise ValueError('registered G0 family does not match frozen seed plan')
    reg.check_usable(fam)
    plan=G0Plan.load(plan_path); entry=plan.get(mechanism_id)
    efs=[f for f in entry.files if f.role=='evaluator' and f.relpath==recipe.evaluator]
    if len(efs)!=1: raise ValueError('recipe evaluator is not uniquely pinned by G0 plan')
    ev=efs[0]; legacy=Path(legacy_root).expanduser().resolve(); evaluator=(legacy/ev.relpath).resolve()
    if sha256_file(evaluator)!=ev.sha256: raise ValueError('evaluator hash mismatch before execution')
    out=Path(output_root).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True); run_dir=out/f'{mechanism_id}_{seed["name"]}'
    if run_dir.exists(): raise FileExistsError(f'G0 output exists: {run_dir}')
    run_dir.mkdir(); raw=run_dir/'raw_evaluator'; raw.mkdir(); before_meta=_tree_metadata(legacy)
    bootstrap=run_dir/'bootstrap.py'; bootstrap.write_text(_bootstrap_source(evaluator,recipe.seed_symbol,seed['start'],seed['stop'],recipe.seed_container,raw))
    env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['PYTHONHASHSEED']='0'
    proc=subprocess.run([sys.executable,'-B',str(bootstrap)],cwd=legacy,env=env,text=True,capture_output=True)
    (run_dir/'stdout.txt').write_text(proc.stdout); (run_dir/'stderr.txt').write_text(proc.stderr)
    after_meta=_tree_metadata(legacy); post=plan.verify({'wsl-canonical':legacy},mechanism_id)[0]; result_path=raw/'result.json'
    raw_result=json.loads(result_path.read_text()) if result_path.is_file() else None
    try: passed=bool(_nested(raw_result,recipe.pass_field)) if raw_result is not None else False
    except KeyError: passed=False
    immutable=(before_meta==after_meta and post.ok and sha256_file(evaluator)==ev.sha256)
    status=ResultStatus.PASS if (proc.returncode==0 and raw_result is not None and passed and immutable) else ResultStatus.FAIL
    report={'format':'cb-g0-run/1','item':'G0','mechanism_id':mechanism_id,'status':status.value,'prereg_id':preg.id,
      'fresh_seed_family':fam.as_dict(),'historical_test_reopened':False,'evaluator_relpath':ev.relpath,'evaluator_sha256':ev.sha256,
      'seed_symbol_overridden':recipe.seed_symbol,'threshold_or_intervention_override_performed':False,
      'raw_legacy_pass_field':recipe.pass_field,'raw_legacy_pass_value':None if raw_result is None else passed,
      'process_returncode':proc.returncode,'legacy_tree_metadata_unchanged':before_meta==after_meta,'pinned_evidence_still_matches':post.ok,
      'raw_result_sha256':sha256_file(result_path) if result_path.is_file() else None,'stdout_sha256':sha256_file(run_dir/'stdout.txt'),
      'stderr_sha256':sha256_file(run_dir/'stderr.txt'),'note':'G0 PASS means isolated fresh-DEV requalification under the frozen evaluator checks. It is not a reopened historical TEST and not an integration/admission claim.'}
    (run_dir/'g0_result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); return report
