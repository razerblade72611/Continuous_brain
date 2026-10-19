"""Independent, CPU-only frozen-core diagnostic; never updates a core or an individual on disk."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch

torch.set_num_threads(1)
WEIGHTS = {
    "learning": "6a7996b21f235d3d52ddd505b49a8b301a04abd5a98ce3af4b895f6b13653f8e",
    "reference": "1b06c3d3e24ba77503d9f304cc4cd01d516c7492fa02d81dc89aa3069b51f780",
    "frozen": "8cffde798661d2661c469fa208799561008d03ae860d2d9be7f1c514a3c6853b",
}


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_sha(core):
    h = hashlib.sha256()
    for key,value in sorted(core.state_dict().items()):
        h.update(key.encode())
        h.update(str((tuple(value.shape),str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def crop_xy(arr, pos, radius=8, fill=-1):
    """The memory and archived map are x-major, unlike pixel frames and the vision's y-major view."""
    d = 2 * radius + 1
    out = np.full((d, d), fill, arr.dtype)
    x0, y0 = int(pos[0]) - radius, int(pos[1]) - radius
    a, b = max(0, x0), max(0, y0)
    c, e = min(arr.shape[0], x0 + d), min(arr.shape[1], y0 + d)
    if c > a and e > b:
        out[a-x0:c-x0, b-y0:e-y0] = arr[a:c, b:e]
    return out


def oov_xy(radius=8):
    out = np.ones((2 * radius + 1, 2 * radius + 1), bool)
    out[radius-4:radius+5, radius-3:radius+4] = False
    return out


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise RuntimeError("CPU wall-time budget reached; preserving partial evidence, no automatic rerun")


def setup(args):
    sys.path.insert(0, str(pathlib.Path(args.code)))
    sys.path.insert(0, str(pathlib.Path(args.code) / "scripts/explore"))
    from continuous_brain_cb2.world_model import PredictiveCore
    cores = {}
    for name, expected in WEIGHTS.items():
        path = pathlib.Path(args.inputs) / name / "cb2-ind-0001" / "weights" / f"{expected}.pt"
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"{name} source weight checksum mismatch")
        pack = torch.load(path, map_location="cpu", weights_only=False)
        core = PredictiveCore.from_state(pack["core"])
        core.eval()
        if not all(torch.isfinite(p).all() for p in core.parameters()):
            raise RuntimeError(f"{name} core has invalid parameters")
        cores[name] = core
    return cores


def collect(seed, args, deadline):
    from cb2_evaluate import load_individual
    from cb2_collect import vision
    from continuous_brain_individual import codec
    from continuous_brain_individual.body import read_body
    from continuous_brain_individual.individual import first_info
    from continuous_brain_cb2.data import body_vector
    from continuous_brain_world.env import W1Env, W1Config
    home = pathlib.Path(args.inputs) / "frozen/cb2-ind-0001"
    ind, meta = load_individual(home, vision())
    before = ind.state_hash()
    if before != meta["state_hash"]:
        raise RuntimeError("frozen source did not restore exactly")
    ind.life = 700_000_000 + seed
    env = W1Env(W1Config())
    obs = env.reset(seed)
    info = first_info(env)
    rows = {k: [] for k in ("image", "body", "action", "map", "seen_age", "step", "origin")}
    done = False
    begin = time.monotonic()
    for i in range(args.max_steps):
        check_deadline(deadline)
        action, utterance = ind.decide(obs, info)
        body = read_body(ind.rt.store.read("driver.body"))
        memory = codec.unblob(ind.rt.store.read("episodic_memory.state"))
        step = int(body["step"])
        pos = body["player_pos"]
        inv = [body["inventory"][k] for k in __import__("continuous_brain_cb2.archive", fromlist=["INVENTORY"]).INVENTORY]
        rows["image"].append(np.array(obs["image"], dtype=np.uint8))
        rows["body"].append(body_vector(np.array(inv)[None], [body["sleeping"]], [body["facing"]], [step], [pos])[0])
        rows["action"].append(action)
        rows["map"].append(crop_xy(memory["map"], pos))
        seen = crop_xy(memory["seen"], pos)
        rows["seen_age"].append(np.where(seen >= 0, step - seen, -1).astype(np.int32))
        rows["step"].append(step)
        last = getattr(ind.rt.components["skills"], "last", None) or {}
        rows["origin"].append(json.dumps(last.get("delegation", ind.read_state("skills").get("delegation", {})), sort_keys=True))
        if i == 0 and (step != 0 or (rows["map"][-1][oov_xy()] >= 0).any()):
            raise RuntimeError("new life did not reset its perceptual map; refuse inherited world targets")
        obs, _, done, info = env.step(action, utterance)
        if done:
            break
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["prev_action"] = np.r_[0, out["action"][:-1]].astype(np.int64)
    out["first"] = np.r_[1.0, np.zeros(len(out["action"])-1)].astype(np.float32)
    out["query"] = np.arange(args.skip, len(out["action"]), args.every, dtype=np.int64)
    duration = time.monotonic() - begin
    return out, {"seed": seed, "rows": len(out["action"]), "queries": len(out["query"]), "died": bool(done),
                 "source_initial_hash": before, "seconds_collect": duration, "scripted_primitive_share": 1.0}


@torch.no_grad()
def features(core, data, args, deadline):
    """Embed once; full history resets only at real starts. One-step keeps the same preceding action."""
    q = data["query"]
    qset = set(q.tolist())
    recurrent, embed_q = [], []
    state = None
    start = time.monotonic()
    for a in range(0, len(data["action"]), args.chunk):
        check_deadline(deadline)
        b = min(len(data["action"]), a + args.chunk)
        emb = core.embed(data["image"][a:b], data["body"][a:b])
        for j, t in enumerate(range(a, b)):
            state, _ = core.rssm.obs_step(None if state is None else dict(state),
                                         core._onehot(data["prev_action"][t:t+1]), emb[j:j+1],
                                         core._t(data["first"][t:t+1]), sample=False)
            if t in qset:
                recurrent.append(core.feat(state)[0].cpu().numpy())
                embed_q.append(emb[j].cpu().numpy())
    if not len(q):
        return None, {"seconds_features": time.monotonic()-start, "queries": 0}
    emb = torch.as_tensor(np.asarray(embed_q))
    one, map_full, map_one = [], [], []
    full = np.asarray(recurrent, np.float32)
    for a in range(0, len(q), args.chunk):
        check_deadline(deadline)
        sl = slice(a, a + args.chunk)
        ids = q[sl]
        post, _ = core.rssm.obs_step(None, core._onehot(data["prev_action"][ids]), emb[sl],
                                    core._t(data["first"][ids]), sample=False)
        one.append(core.feat(post).cpu().numpy())
        if "map" in core.heads:
            map_one.append(core.map_logits(core.feat(post)).argmax(-1).cpu().numpy())
            map_full.append(core.map_logits(torch.as_tensor(full[sl])).argmax(-1).cpu().numpy())
    one = np.concatenate(one)
    frame = np.c_[np.asarray(embed_q), np.eye(17, dtype=np.float32)[data["prev_action"][q]]]
    out = {"full": full, "one": one, "frame": frame}
    if map_full:
        if core.cfg.map_radius != 8:
            raise RuntimeError("secondary map crop requires radius 8; do not reinterpret other geometry")
        out["map_full"] = np.concatenate(map_full)
        out["map_one"] = np.concatenate(map_one)
    if not all(np.isfinite(v).all() for v in out.values()):
        raise RuntimeError("non-finite inference output")
    return out, {"seconds_features": time.monotonic()-start, "queries": len(q),
                 "widths": {k: v.shape[-1] for k, v in out.items()}, "map_radius": core.cfg.map_radius}


def self_check(args):
    sys.path.insert(0, str(pathlib.Path(args.code)))
    from continuous_brain_cb2.world_model import PredictiveCore, CoreConfig
    from continuous_brain_individual.memory import view_cells
    from continuous_brain_cb2.record import map_crop
    pos = (31, 31)
    mask = oov_xy()
    actual = np.ones((17, 17), bool)
    for _, _, x, y in view_cells(pos):
        actual[x-pos[0]+8, y-pos[1]+8] = False
    assert np.array_equal(mask, actual)
    arr = np.arange(64*64, dtype=np.int32).reshape(64, 64)
    ours = crop_xy(arr, pos)
    stored = map_crop({"map": arr}, pos).reshape(33, 33)[8:25, 8:25]
    assert np.array_equal(ours.astype(np.int8), stored)
    core = PredictiveCore(CoreConfig.sized("XS"))
    core.cfg.map_radius = 8
    old = core.out_of_view_mask().reshape(17, 17)
    # The old y/x mask leaks currently visible cells and excludes genuinely hidden ones.
    report = {"crop_matches_archive_xy": True, "correct_oov_matches_actual_view": True,
              "old_mask_visible_cells_in_oov": int((old & ~actual).sum()),
              "old_mask_hidden_cells_excluded": int((~old & actual).sum())}
    return report


def pilot(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pilot.json").exists():
        raise RuntimeError("pilot already recorded; no automatic overwrite or rerun")
    started = time.monotonic()
    deadline = started + 2700
    cores = setup(args)
    records = []
    for seed in (83090, 83091):
        data, record = collect(seed, args, deadline)
        for name, core in cores.items():
            _, elapsed = features(core, data, args, deadline)
            record[name] = elapsed
        np.savez_compressed(out / f"pilot-input-{seed}.npz", **data)
        record["sha256"] = sha(out / f"pilot-input-{seed}.npz")
        records.append(record)
        print(json.dumps({"pilot_seed": seed, "rows": record["rows"], "seconds": record["seconds_collect"] + sum(record[n]["seconds_features"] for n in cores)}), flush=True)
    per_step = max((r["seconds_collect"] + sum(r[n]["seconds_features"] for n in cores))/r["rows"] for r in records)
    total = time.monotonic() - started
    estimates = {str(n): total + 1.25 * n * args.max_steps * per_step + 360 for n in (48, 96)}
    eligible = [n for n in (48, 96) if estimates[str(n)] <= 2700]
    report = {"format": "sol-memory-timing/1", "script_sha256": sha(__file__), "seed_block": [83090,83091],
              "records": records, "elapsed": total, "worst_observed_seconds_per_step": per_step,
              "conservative_seconds_estimates": estimates, "recommended_N": max(eligible) if eligible else None,
              "selection_uses": "timing and row counts only; pilot prediction scores are never computed",
              "weights": WEIGHTS, "geometry": self_check(args)}
    save_json(out / "pilot.json", report)
    print(json.dumps(report, indent=2), flush=True)


def query_targets(data):
    q = data["query"]
    target = (data["map"][q] >= 0)
    age = data["seen_age"][q]
    xy = np.rint(data["body"][q, -2:] * 64).astype(int)
    dx = np.arange(-8, 9)[:, None]
    dy = np.arange(-8, 9)[None, :]
    valid = np.stack([(oov_xy() & (x+dx >= 0) & (x+dx < 64) & (y+dy >= 0) & (y+dy < 64)) for x,y in xy]) if len(q) else np.empty((0,17,17),bool)
    return target.reshape(len(q),289), age.reshape(len(q),289), valid.reshape(len(q),289)


def balanced(y, pred, valid):
    pos = valid & y
    neg = valid & ~y
    if not pos.any() or not neg.any():
        return None
    return float(0.5 * (pred[pos].mean() + (~pred[neg]).mean()))


def auc(y, raw, valid):
    from scipy.stats import rankdata
    truth, score = y[valid], raw[valid]
    npos, nneg = int(truth.sum()), int((~truth).sum())
    if not npos or not nneg:
        return None
    ranks = rankdata(score)
    return float((ranks[truth].sum() - npos*(npos+1)/2)/(npos*nneg))


def summarize_prediction(data, raw):
    y, age, valid = query_targets(data)
    pred = raw >= 0.5
    result = {"balanced_accuracy": balanced(y,pred,valid), "auroc": auc(y,raw,valid),
              "eligible_queries": len(data["query"]), "positive_tiles": int((y&valid).sum()),
              "negative_tiles": int((~y&valid).sum()), "linear_values_outside_0_1": float(((raw<0)|(raw>1))[valid].mean()) if valid.any() else None,
              "age": {}}
    for name, ages in (("1-10",(age>=1)&(age<=10)), ("11-50",(age>=11)&(age<=50)), ("51+",age>=51)):
        # Each age stratum uses its positive tiles and the SAME unknown negatives, explicitly reported.
        mask = valid & ((y & ages) | ~y)
        result["age"][name] = {"balanced_accuracy": balanced(y,pred,mask), "auroc": auc(y,raw,mask),
                              "positive_tiles":int((mask&y).sum()), "negative_tiles":int((mask&~y).sum())}
    tile_scores = [balanced(y[:,j],pred[:,j],valid[:,j]) for j in range(289)]
    present = [x for x in tile_scores if x is not None]
    result["macro_offset_balanced"] = float(np.mean(present)) if present else None
    result["offsets_with_both_classes"] = len(present)
    return result


def fit_ridge(X,Y,lam):
    """Fit-only normalization and a centered, regularized linear decoder. No validation/test normalization."""
    from scipy.linalg import solve
    mu, sd, ym = X.mean(0), X.std(0), Y.mean(0)
    sd = np.where(sd>1e-6,sd,1.0)
    z = np.asarray((X-mu)/sd,np.float64)
    target = np.asarray(Y-ym,np.float64)
    if len(z) <= z.shape[1]:
        gram = z @ z.T
        gram.flat[::len(z)+1] += lam*len(z)
        W = z.T @ solve(gram,target,assume_a="pos",check_finite=True)
    else:
        gram = z.T @ z
        gram.flat[::gram.shape[0]+1] += lam*len(z)
        W = solve(gram,z.T@target,assume_a="pos",check_finite=True)
    return mu,sd,ym,W


def predict(model,X):
    mu,sd,ym,W = model
    return ((X-mu)/sd)@W+ym


def paired_summary(scores, first, second):
    from scipy.stats import norm
    pairs = [(seed,row[first]["balanced_accuracy"],row[second]["balanced_accuracy"])
             for seed,row in scores.items() if row[first]["balanced_accuracy"] is not None and row[second]["balanced_accuracy"] is not None]
    if not pairs:
        return {"n":0,"mean":None,"lower95":None,"upper95":None}
    diff = np.asarray([a-b for _,a,b in pairs])
    rng = np.random.default_rng(20261011)
    draws = diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
    sd = float(diff.std(ddof=1)) if len(diff)>1 else None
    return {"n":len(diff),"worlds":[seed for seed,_,_ in pairs],"mean":float(diff.mean()),
            "lower95":float(np.quantile(draws,0.05)),"upper95":float(np.quantile(draws,0.95)),
            "paired_sd":sd,"normal_approx_MDE80":float((norm.ppf(.95)+norm.ppf(.8))*sd/np.sqrt(len(diff))) if sd is not None else None}


def run(args):
    plan_path = pathlib.Path(args.plan)
    plan = json.loads(plan_path.read_text())
    if (plan["status"] != "LOCKED" or sha(__file__) != plan["script_sha256"]
            or any(getattr(args,key) != plan[key] for key in ("max_steps","skip","every","chunk"))
            or plan["weights"] != WEIGHTS):
        raise RuntimeError("locked plan/script/settings mismatch")
    out = pathlib.Path(args.out)
    if out.exists():
        raise RuntimeError("main output already exists; refuse overwrite/rerun")
    out.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + plan["budget_seconds"]
    cores = setup(args)
    before_models = {name:model_sha(core) for name,core in cores.items()}
    frozen_meta = pathlib.Path(args.inputs)/"frozen/cb2-ind-0001/individual.json"
    before_meta = sha(frozen_meta)
    if sha(pathlib.Path(args.code)/"runs/explore/vision/tile_decoder_v1.pt") != plan["vision_sha256"]:
        raise RuntimeError("vision source differs from the locked plan")
    save_json(out/"run.json",{"plan_sha256":sha(plan_path),"plan":plan,"weights":WEIGHTS,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    inputs, all_features, episodes = {},{},[]
    with (out/"progress.jsonl").open("a",buffering=1) as log:
        for seed in plan["fit"]+plan["validation"]+plan["test"]:
            data, record = collect(seed,args,deadline)
            np.savez_compressed(out/f"input-{seed}.npz",**data)
            record["input_sha256"] = sha(out/f"input-{seed}.npz")
            inputs[seed] = data
            all_features[seed] = {}
            for name,core in cores.items():
                feat, elapsed = features(core,data,args,deadline)
                record[name] = elapsed
                if feat is not None:
                    np.savez_compressed(out/f"features-{name}-{seed}.npz",**feat)
                    record[name]["sha256"] = sha(out/f"features-{name}-{seed}.npz")
                    all_features[seed][name] = feat
            episodes.append(record)
            progress = {"stage":"collected", "seed":seed,"completed":len(episodes),"total":plan["N"],"rows":record["rows"],"elapsed":time.monotonic()-started}
            log.write(json.dumps(progress)+"\n")
            print(json.dumps(progress),flush=True)
        save_json(out/"episodes.json",episodes)
        fit_seeds = [s for s in plan["fit"] if len(inputs[s]["query"])]
        val_seeds = [s for s in plan["validation"] if len(inputs[s]["query"])]
        test_seeds = [s for s in plan["test"] if len(inputs[s]["query"])]
        if not fit_seeds or not val_seeds:
            raise RuntimeError("no eligible fitting or validation data; report incomplete, do not replace worlds")
        Y = np.concatenate([query_targets(inputs[s])[0].astype(np.float64) for s in fit_seeds])
        prior = Y.mean(0)
        scores = {s:{"constant":summarize_prediction(inputs[s],np.tile(prior,(len(inputs[s]["query"]),1)))} for s in test_seeds}
        models = {}
        # Test predictions are made only after each condition's regularization is selected on the validation worlds.
        for name in cores:
            for kind in ("full","one","frame"):
                check_deadline(deadline)
                X = np.concatenate([all_features[s][name][kind] for s in fit_seeds])
                candidates = []
                for lam in (100.,10.,1.,.1,.01):
                    check_deadline(deadline)
                    model = fit_ridge(X,Y,lam)
                    val = [summarize_prediction(inputs[s],predict(model,all_features[s][name][kind]))["balanced_accuracy"] for s in val_seeds]
                    valid = [v for v in val if v is not None]
                    if len(valid) < 4:
                        raise RuntimeError("fewer than four validation worlds with both classes; preserve and report incomplete")
                    candidates.append((float(np.mean(valid)),lam,model))
                # Exact ties select the largest penalty, the first candidate in the declared order.
                selected = max(candidates,key=lambda c:c[0])
                key = f"{name}/{kind}"
                models[key] = {"penalty":selected[1],"validation_world_mean_balanced":selected[0],"feature_width":X.shape[1],"fit_rows":len(X)}
                np.savez_compressed(out/f"readout-{name}-{kind}.npz",mu=selected[2][0],sd=selected[2][1],ym=selected[2][2],W=selected[2][3])
                for s in test_seeds:
                    raw = predict(selected[2],all_features[s][name][kind])
                    scores[s][key] = summarize_prediction(inputs[s],raw)
                log.write(json.dumps({"stage":"readout_scored","condition":key,"penalty":selected[1],"elapsed":time.monotonic()-started})+"\n")
                print("SCORED",key,"penalty",selected[1],flush=True)
        contrasts = {}
        for name in cores:
            for kind in ("one","frame"):
                contrasts[f"{name}/full-minus-{kind}"] = paired_summary(scores,f"{name}/full",f"{name}/{kind}")
        for other in ("reference","frozen","constant"):
            contrasts[f"learning/full-minus-{other}"] = paired_summary(scores,"learning/full",f"{other}/full" if other!="constant" else "constant")
        map_heads = {}
        for name in ("learning","reference"):
            results = []
            for s in test_seeds:
                target = inputs[s]["map"][inputs[s]["query"]].reshape(-1,289)+1
                _,_,mask = query_targets(inputs[s])
                valid = mask & (target>0)
                f = all_features[s][name]
                if not valid.any():
                    results.append({"seed":s,"full_balanced":None,"one_balanced":None,"seen_tiles":0})
                    continue
                classes = np.unique(target[valid])
                metric = lambda p:float(np.mean([(p[valid][target[valid]==c]==c).mean() for c in classes]))
                results.append({"seed":s,"full_balanced":metric(f["map_full"]),"one_balanced":metric(f["map_one"]),"seen_tiles":int(valid.sum()),"classes":classes.tolist()})
            map_heads[name]=results
        required = ("learning/full-minus-one","learning/full-minus-frame","learning/full-minus-reference","learning/full-minus-frozen","learning/full-minus-constant")
        enough = all(contrasts[k]["n"] >= 12 for k in required)
        evidence = enough and all(contrasts[k]["lower95"]>0 for k in required)
        history = contrasts["learning/full-minus-one"]
        if not enough:
            verdict = "INCONCLUSIVE: fewer than 12 eligible test worlds"
        elif evidence:
            verdict = "HISTORY_AND_EXPERIENCE_ADVANTAGE_SHOWN_ON_THIS_DIAGNOSTIC"
        elif history["upper95"] < .02:
            verdict = "NO_HISTORY_ADVANTAGE_OF_TWO_PERCENTAGE_POINTS_SUPPORTED_HERE"
        else:
            verdict = "INCONCLUSIVE_FOR_JOINT_HISTORY_AND_EXPERIENCE_CLAIM"
        after_models = {name:model_sha(core) for name,core in cores.items()}
        if before_models != after_models or sha(frozen_meta) != before_meta:
            raise RuntimeError("frozen input/model changed during the diagnostic")
        report = {"format":"sol-memory-diagnostic/1","plan_sha256":sha(plan_path),"script_sha256":sha(__file__),
                  "weights":WEIGHTS,"episode_counts":{"fit":len(plan["fit"]),"validation":len(plan["validation"]),"test":len(plan["test"]),"test_eligible":len(test_seeds)},
                  "episodes":episodes,"readouts":models,"test_by_world":scores,"contrasts":contrasts,"map_heads":map_heads,
                  "verdict":verdict,"elapsed_seconds":time.monotonic()-started,
                  "limitations":["S2b policy distribution; no inference about bias direction","engineered perceptual map target, not general memory","earlier seed exposure not exhaustively proven","CPU diagnostic, no individual core updates or behavioral growth claim","age-stratum unknown negatives are reused across strata","world bootstrap conditions on fitted readouts; it does not cover all fitting-data variability"],
                  "learned_primitive_action_share":0.0,"geometry":self_check(args),
                  "core_parameters_before":before_models,"core_parameters_after":after_models,
                  "source_metadata_sha256_before_after":before_meta}
        save_json(out/"result.json",report)
        print("VERDICT",verdict,"SECONDS",report["elapsed_seconds"],flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("self-check", "pilot", "run"))
    p.add_argument("--code", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--skip", type=int, default=25)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--plan")
    args = p.parse_args()
    if args.mode == "self-check":
        print(json.dumps(self_check(args), indent=2))
    elif args.mode == "pilot":
        pilot(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
