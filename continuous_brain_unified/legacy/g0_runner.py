"""G0 isolated legacy requalification runner.

v0.38.1.dev3 adds two fail-closed layers discovered by the first real G0 run:
1. historical core overlay: first-wave evaluators ran before later mutations to the
   accumulated v0.12.2 source tree, so protected historical checkpoints are loaded
   with the original v0.12.2 core modules while mechanism/evaluator files remain
   hash-pinned to their historical overlay versions;
2. engineering-only runtime preflight: a full evaluator execution on a disjoint
   ENGINEERING family must complete and produce result.json before a scientific
   DEV family can be opened. Preflight metrics/pass fields are never interpreted.
"""
from __future__ import annotations
import ast, json, os, subprocess, sys, tempfile, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from continuous_brain_protocols import ResultStatus, SeedPurpose, sha256_file, sha256_json
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_unified.prereg import PreregistrationRegistry, PreregistrationError
from .g0 import G0Plan

RECIPE_FORMAT='cb-g0-recipes/1'
SEED_FORMAT='cb-g0-seed-plan/1'
PREFLIGHT_SEED_FORMAT='cb-g0-preflight-seed-plan/1'
CORE_FORMAT='cb-historical-core-overlay/1'

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

def load_preflight_seed_plan(path='legacy/g0_preflight_seed_plan.json'):
    raw=_load_json(path)
    if raw.get('format')!=PREFLIGHT_SEED_FORMAT: raise ValueError('bad G0 preflight seed-plan format')
    return {f['mechanism_id']:f for f in raw['families']}

def load_dependency_closure(path='legacy/g0_dependency_closure.json'):
    raw=_load_json(path)
    if raw.get('format')!='cb-g0-dependency-closure/1': raise ValueError('bad G0 dependency-closure format')
    return raw['mechanisms']

def load_historical_core(path='legacy/historical_core_v0122.json'):
    raw=_load_json(path)
    if raw.get('format')!=CORE_FORMAT: raise ValueError('bad historical-core overlay format')
    return raw

def _verify_dependency_closure(legacy:Path, mechanism_id:str, path='legacy/g0_dependency_closure.json') -> tuple[bool,list[dict]]:
    closures=load_dependency_closure(path); expected=closures.get(mechanism_id)
    if not expected: return False,[{'relpath':None,'error':'dependency closure missing'}]
    problems=[]
    for f in expected:
        p=(legacy/f['relpath']).resolve()
        if legacy not in p.parents and p!=legacy:
            problems.append({'relpath':f['relpath'],'error':'resolves outside legacy root'}); continue
        if not p.is_file():
            problems.append({'relpath':f['relpath'],'error':'missing'}); continue
        got=sha256_file(p)
        if got!=f['sha256']:
            problems.append({'relpath':f['relpath'],'expected':f['sha256'],'actual':got})
    return not problems,problems

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

def _historical_archive_path(value:str|Path|None)->Path|None:
    if value is None:
        value=os.environ.get('CONTINUOUS_BRAIN_BASELINE_ARCHIVE')
    if not value: return None
    return Path(value).expanduser().resolve()

def verify_historical_core_archive(archive:str|Path|None, manifest_path='legacy/historical_core_v0122.json')->tuple[bool,list[dict]]:
    p=_historical_archive_path(archive); spec=load_historical_core(manifest_path); problems=[]
    if p is None:
        return False,[{'error':'historical core archive not provided; use --historical-core-archive or CONTINUOUS_BRAIN_BASELINE_ARCHIVE'}]
    if not p.is_file(): return False,[{'archive':str(p),'error':'missing'}]
    got=sha256_file(p)
    if got!=spec['archive_sha256']:
        problems.append({'archive':str(p),'expected':spec['archive_sha256'],'actual':got})
        return False,problems
    try:
        with zipfile.ZipFile(p) as z:
            names=set(z.namelist()); prefix=spec['archive_prefix']
            import hashlib
            for f in spec['files']:
                member=prefix+f['relpath']
                if member not in names:
                    problems.append({'relpath':f['relpath'],'error':'missing from historical archive'}); continue
                h=hashlib.sha256(z.read(member)).hexdigest()
                if h!=f['sha256']: problems.append({'relpath':f['relpath'],'expected':f['sha256'],'actual':h})
    except zipfile.BadZipFile as exc:
        problems.append({'archive':str(p),'error':f'bad zip: {exc}'})
    return not problems,problems

def _extract_historical_core(archive:str|Path, dest:Path, manifest_path='legacy/historical_core_v0122.json')->dict[str,str]:
    p=_historical_archive_path(archive); assert p is not None
    ok,problems=verify_historical_core_archive(p,manifest_path)
    if not ok: raise RuntimeError('historical core archive mismatch: '+json.dumps(problems,sort_keys=True))
    spec=load_historical_core(manifest_path); prefix=spec['archive_prefix']; dest.mkdir(parents=True,exist_ok=True); modules={}
    with zipfile.ZipFile(p) as z:
        for f in spec['files']:
            rel=f['relpath']; out=dest/rel; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(z.read(prefix+rel))
            if sha256_file(out)!=f['sha256']: raise RuntimeError('extracted historical core hash mismatch: '+rel)
            if rel.endswith('.py'):
                mod=rel[:-3].replace('/','.')
                modules[mod]=str(out.resolve())
    return modules

def _historical_core_digest(manifest_path='legacy/historical_core_v0122.json')->str:
    return sha256_json(load_historical_core(manifest_path))

def prepare(mechanism_id:str, legacy_root:str|Path, plan_path='legacy/g0_plan.json', registry_dir=None, historical_core_archive=None)->G0Prepared:
    legacy=Path(legacy_root).expanduser().resolve()
    plan=G0Plan.load(plan_path); v=plan.verify({'wsl-canonical':legacy},mechanism_id)[0]
    recipes,blocked=load_recipes(); seeds=load_seed_plan(); recipe=recipes.get(mechanism_id); reasons=[]
    if not v.ok: reasons.append('pinned G0 evidence hashes do not match')
    if mechanism_id in blocked: reasons.append(blocked[mechanism_id])
    seed=None if recipe is None else seeds.get(recipe.fresh_family)
    if recipe is not None and seed is None: reasons.append('recipe seed family missing from approved plan')
    if recipe is not None and v.ok:
        evaluator=legacy/recipe.evaluator
        reasons.extend(_static_evaluator_preflight(evaluator,recipe.seed_symbol))
        deps_ok,dep_problems=_verify_dependency_closure(legacy,mechanism_id)
        if not deps_ok: reasons.append('project-local evaluator dependency closure mismatch: '+json.dumps(dep_problems,sort_keys=True))
        core_ok,core_problems=verify_historical_core_archive(historical_core_archive)
        if not core_ok: reasons.append('historical core overlay unavailable/mismatched: '+json.dumps(core_problems,sort_keys=True))
    preg=PreregistrationRegistry.load(registry_dir).entries.get('G0-LEGACY-REQUAL-v1')
    return G0Prepared(mechanism_id,v.ok,'MISSING' if preg is None else preg.status,recipe,seed,tuple(reasons))

def _static_evaluator_preflight(evaluator: Path, seed_symbol: str) -> list[str]:
    """Read-only syntax/API preflight. Does not import or execute historical code."""
    problems=[]
    try: tree=ast.parse(evaluator.read_text(), filename=str(evaluator))
    except Exception as exc: return [f"evaluator AST parse failed: {exc}"]
    assigned=set(); functions=set(); state_dir_literal=False
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t,ast.Name): assigned.add(t.id)
        elif isinstance(node,ast.ImportFrom):
            for a in node.names: assigned.add(a.asname or a.name)
        elif isinstance(node,ast.Import):
            for a in node.names: assigned.add(a.asname or a.name.split('.')[0])
        elif isinstance(node,ast.FunctionDef): functions.add(node.name)
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='--state-dir': state_dir_literal=True
    if seed_symbol not in assigned: problems.append(f"seed symbol {seed_symbol!r} is not assigned at module level")
    if 'main' not in functions: problems.append('evaluator has no main() function')
    if not state_dir_literal: problems.append('evaluator source does not contain --state-dir')
    return problems

def _bootstrap_source(evaluator:Path, seed_symbol:str, start:int, stop:int, container:str, state_dir:Path, overrides:dict[str,str])->str:
    # Import hook is installed before continuous_brain is imported. Only the exact
    # hash-verified historical core modules in `overrides` are redirected.
    return f'''import importlib.abc, importlib.util, pathlib, sys\nsys.dont_write_bytecode=True\nOVERRIDES={overrides!r}\nclass _HistoricalCoreFinder(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        p=OVERRIDES.get(fullname)\n        if p is None: return None\n        return importlib.util.spec_from_file_location(fullname,p)\nsys.meta_path.insert(0,_HistoricalCoreFinder())\nroot=pathlib.Path({str(evaluator.parent)!r}).resolve()\nsys.path.insert(0,str(root))\nspec=importlib.util.spec_from_file_location("cb_g0_frozen_evaluator", {str(evaluator)!r})\nm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\nvals=range({start},{stop+1})\nsetattr(m,{seed_symbol!r}, list(vals) if {container!r}=="list" else range({start},{stop+1}))\nsys.argv=[{str(evaluator)!r},"--state-dir",{str(state_dir)!r}]\nm.main()\n'''

def _spent_dict(fam)->dict:
    d=fam.as_dict(); d['spent']=True; return d

def _preflight_attestation_path(directory:str|Path|None, mechanism_id:str)->Path:
    d=Path(directory or 'evidence/g0_runtime_preflight'); return d/f'{mechanism_id}.json'

def runtime_preflight(mechanism_id:str, legacy_root:str|Path, output_root:str|Path='runs/g0_runtime_preflight', *, registry_dir=None, plan_path='legacy/g0_plan.json', historical_core_archive=None, attestation_dir='evidence/g0_runtime_preflight')->dict[str,Any]:
    prepared=prepare(mechanism_id,legacy_root,plan_path,registry_dir,historical_core_archive)
    if prepared.blocked_reasons:
        return {'format':'cb-g0-runtime-preflight/1','item':'G0_RUNTIME_PREFLIGHT','mechanism_id':mechanism_id,'status':ResultStatus.BLOCKED.value,'reasons':list(prepared.blocked_reasons)}
    recipe=prepared.recipe; assert recipe
    pf=load_preflight_seed_plan().get(mechanism_id)
    if pf is None: return {'format':'cb-g0-runtime-preflight/1','item':'G0_RUNTIME_PREFLIGHT','mechanism_id':mechanism_id,'status':ResultStatus.BLOCKED.value,'reasons':['engineering preflight family missing']}
    reg=SeedRegistry.load(registry_dir); fam=reg.get(pf['name']); reg.check_usable(fam)
    legacy=Path(legacy_root).expanduser().resolve(); plan=G0Plan.load(plan_path); entry=plan.get(mechanism_id)
    efs=[f for f in entry.files if f.role=='evaluator' and f.relpath==recipe.evaluator]
    if len(efs)!=1: raise ValueError('recipe evaluator is not uniquely pinned by G0 plan')
    ev=efs[0]; evaluator=(legacy/ev.relpath).resolve()
    out=Path(output_root).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True); run_dir=out/f'{mechanism_id}_{pf["name"]}'
    if run_dir.exists(): raise FileExistsError(f'G0 preflight output exists: {run_dir}')
    run_dir.mkdir(); raw=run_dir/'raw_evaluator'; raw.mkdir(); core_dir=run_dir/'historical_core'; overrides=_extract_historical_core(historical_core_archive,core_dir)
    before_meta=_tree_metadata(legacy); reg.spend(fam,source=f'v0.38.1.dev3:G0_RUNTIME_PREFLIGHT:{mechanism_id}:opened')
    bootstrap=run_dir/'bootstrap.py'; bootstrap.write_text(_bootstrap_source(evaluator,recipe.seed_symbol,pf['start'],pf['stop'],recipe.seed_container,raw,overrides))
    env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['PYTHONHASHSEED']='0'
    proc=subprocess.run([sys.executable,'-B',str(bootstrap)],cwd=legacy,env=env,text=True,capture_output=True)
    (run_dir/'stdout.txt').write_text(proc.stdout); (run_dir/'stderr.txt').write_text(proc.stderr)
    after_meta=_tree_metadata(legacy); post=plan.verify({'wsl-canonical':legacy},mechanism_id)[0]; post_deps_ok,post_dep_problems=_verify_dependency_closure(legacy,mechanism_id); result_path=raw/'result.json'
    completed=result_path.is_file(); immutable=(before_meta==after_meta and post.ok and post_deps_ok and sha256_file(evaluator)==ev.sha256)
    ok=bool(completed and immutable)
    report={'format':'cb-g0-runtime-preflight/1','item':'G0_RUNTIME_PREFLIGHT','mechanism_id':mechanism_id,
      'status':ResultStatus.ENGINEERING_ONLY.value if ok else ResultStatus.INFRASTRUCTURE_FAILURE.value,
      'scientific_interpretation_permitted':False,'runtime_completed':completed,'process_returncode':proc.returncode,
      'engineering_seed_family':_spent_dict(fam),'historical_test_reopened':False,'scientific_dev_family_opened':False,
      'historical_core_archive_sha256':load_historical_core()['archive_sha256'],'historical_core_manifest_digest':_historical_core_digest(),
      'evaluator_relpath':ev.relpath,'evaluator_sha256':ev.sha256,'legacy_tree_metadata_unchanged':before_meta==after_meta,
      'pinned_evidence_still_matches':post.ok,'dependency_closure_still_matches':post_deps_ok,'dependency_closure_post_problems':post_dep_problems,
      'raw_result_exists':completed,'raw_result_sha256':sha256_file(result_path) if completed else None,
      'stdout_sha256':sha256_file(run_dir/'stdout.txt'),'stderr_sha256':sha256_file(run_dir/'stderr.txt'),
      'note':'ENGINEERING_ONLY compatibility check. Evaluator metrics/pass fields are intentionally not copied into this attestation and must not be interpreted.'}
    (run_dir/'preflight_result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    att=_preflight_attestation_path(attestation_dir,mechanism_id); att.parent.mkdir(parents=True,exist_ok=True); att.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

def _require_runtime_preflight(mechanism_id:str, evaluator_sha:str, historical_core_archive, attestation_dir='evidence/g0_runtime_preflight')->dict:
    p=_preflight_attestation_path(attestation_dir,mechanism_id)
    if not p.is_file(): raise RuntimeError(f'G0 runtime preflight missing for {mechanism_id}; run --runtime-preflight before opening DEV')
    d=json.loads(p.read_text())
    if d.get('status')!=ResultStatus.ENGINEERING_ONLY.value or d.get('runtime_completed') is not True:
        raise RuntimeError(f'G0 runtime preflight did not complete cleanly for {mechanism_id}')
    if d.get('evaluator_sha256')!=evaluator_sha: raise RuntimeError('G0 runtime preflight evaluator hash no longer matches')
    core=load_historical_core()
    if d.get('historical_core_archive_sha256')!=core['archive_sha256'] or d.get('historical_core_manifest_digest')!=_historical_core_digest():
        raise RuntimeError('G0 runtime preflight historical-core identity no longer matches')
    ok,problems=verify_historical_core_archive(historical_core_archive)
    if not ok: raise RuntimeError('historical core archive mismatch: '+json.dumps(problems,sort_keys=True))
    return d

def run(mechanism_id:str, legacy_root:str|Path, output_root:str|Path, *, registry_dir=None, plan_path='legacy/g0_plan.json', confirm_open:str|None=None, historical_core_archive=None, attestation_dir='evidence/g0_runtime_preflight')->dict[str,Any]:
    prepared=prepare(mechanism_id,legacy_root,plan_path,registry_dir,historical_core_archive)
    if prepared.blocked_reasons:
        return {'item':'G0','mechanism_id':mechanism_id,'status':ResultStatus.BLOCKED.value,'reasons':list(prepared.blocked_reasons),'prereg_status':prepared.prereg_status}
    try: preg=PreregistrationRegistry.load(registry_dir).require('G0-LEGACY-REQUAL-v1',item='G0')
    except PreregistrationError as e: raise PreregistrationError('G0 execution is locked until David approves G0-LEGACY-REQUAL-v1: '+str(e)) from e
    if confirm_open!='G0-LEGACY-REQUAL-v1': raise PreregistrationError('scientific G0 execution requires --confirm-open G0-LEGACY-REQUAL-v1')
    recipe=prepared.recipe; seed=prepared.seed_plan; assert recipe and seed
    reg=SeedRegistry.load(registry_dir); fam=reg.get(seed['name'])
    if fam.purpose!=SeedPurpose.DEV or fam.start!=seed['start'] or fam.stop!=seed['stop']: raise ValueError('registered G0 family does not match frozen seed plan')
    reg.check_usable(fam)
    plan=G0Plan.load(plan_path); entry=plan.get(mechanism_id); efs=[f for f in entry.files if f.role=='evaluator' and f.relpath==recipe.evaluator]
    if len(efs)!=1: raise ValueError('recipe evaluator is not uniquely pinned by G0 plan')
    ev=efs[0]; legacy=Path(legacy_root).expanduser().resolve(); evaluator=(legacy/ev.relpath).resolve()
    if sha256_file(evaluator)!=ev.sha256: raise ValueError('evaluator hash mismatch before execution')
    _require_runtime_preflight(mechanism_id,ev.sha256,historical_core_archive,attestation_dir)
    out=Path(output_root).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True); run_dir=out/f'{mechanism_id}_{seed["name"]}'
    if run_dir.exists(): raise FileExistsError(f'G0 output exists: {run_dir}')
    run_dir.mkdir(); raw=run_dir/'raw_evaluator'; raw.mkdir(); core_dir=run_dir/'historical_core'; overrides=_extract_historical_core(historical_core_archive,core_dir); before_meta=_tree_metadata(legacy)
    reg.spend(fam,source=f'G0-LEGACY-REQUAL-v1:{mechanism_id}:opened')
    spent_fam=_spent_dict(fam)
    (run_dir/'seed_opened.json').write_text(json.dumps({'prereg_id':'G0-LEGACY-REQUAL-v1','mechanism_id':mechanism_id,'family':spent_fam,'spent_before_launch':True},indent=2,sort_keys=True)+'\n')
    bootstrap=run_dir/'bootstrap.py'; bootstrap.write_text(_bootstrap_source(evaluator,recipe.seed_symbol,seed['start'],seed['stop'],recipe.seed_container,raw,overrides))
    env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['PYTHONHASHSEED']='0'
    proc=subprocess.run([sys.executable,'-B',str(bootstrap)],cwd=legacy,env=env,text=True,capture_output=True)
    (run_dir/'stdout.txt').write_text(proc.stdout); (run_dir/'stderr.txt').write_text(proc.stderr)
    after_meta=_tree_metadata(legacy); post=plan.verify({'wsl-canonical':legacy},mechanism_id)[0]; post_deps_ok,post_dep_problems=_verify_dependency_closure(legacy,mechanism_id); result_path=raw/'result.json'; raw_result=json.loads(result_path.read_text()) if result_path.is_file() else None
    try: passed=bool(_nested(raw_result,recipe.pass_field)) if raw_result is not None else False
    except KeyError: passed=False
    immutable=(before_meta==after_meta and post.ok and post_deps_ok and sha256_file(evaluator)==ev.sha256)
    if raw_result is None or not immutable or (passed and proc.returncode!=0): status=ResultStatus.INFRASTRUCTURE_FAILURE
    elif passed: status=ResultStatus.PASS
    else: status=ResultStatus.FAIL
    report={'format':'cb-g0-run/2','item':'G0','mechanism_id':mechanism_id,'status':status.value,'prereg_id':preg.id,
      'fresh_seed_family':spent_fam,'fresh_seed_family_spent_before_launch':True,'historical_test_reopened':False,
      'historical_core_archive_sha256':load_historical_core()['archive_sha256'],'historical_core_manifest_digest':_historical_core_digest(),
      'evaluator_relpath':ev.relpath,'evaluator_sha256':ev.sha256,'seed_symbol_overridden':recipe.seed_symbol,'threshold_or_intervention_override_performed':False,
      'raw_legacy_pass_field':recipe.pass_field,'raw_legacy_pass_value':None if raw_result is None else passed,'process_returncode':proc.returncode,
      'legacy_tree_metadata_unchanged':before_meta==after_meta,'pinned_evidence_still_matches':post.ok,'dependency_closure_still_matches':post_deps_ok,'dependency_closure_post_problems':post_dep_problems,
      'raw_result_sha256':sha256_file(result_path) if result_path.is_file() else None,'stdout_sha256':sha256_file(run_dir/'stdout.txt'),'stderr_sha256':sha256_file(run_dir/'stderr.txt'),
      'note':'G0 PASS/FAIL is scientific only when a raw evaluator result exists and provenance is intact. Missing result/provenance/setup failures are INFRASTRUCTURE_FAILURE, never scientific FAIL.'}
    (run_dir/'g0_result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); return report
