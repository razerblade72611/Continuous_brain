"""Train and evaluate the vision organ's tile decoder (supervised developmental phase, W1 teacher labels).

    /home/claude/venvs/explore/bin/python scripts/explore/train_vision_organ.py

1. **Data.**  W1 episodes on three disjoint seed sets (train / val / test), one fresh world per episode, with a
   behaviour mix (``continuous_brain_organs.vision_data``): the privileged scripted baseline, a teacher-assisted
   explorer (caves, lava, ores, tunnels, monsters, night), a home-bound gardener (saplings and ripe plants) and
   the uniform random baseline.  Every ``stride``-th observation is kept with its teacher labels (the privileged
   material map and objects in the visible window).  Written to ``runs/explore/vision/data_v1/`` (gitignored).
2. **Training.**  ``TileDecoder`` (seeded, ``torch.use_deterministic_algorithms``, 2 threads), AdamW, cosine
   schedule, cross-entropy on both heads, a fixed number of epochs sized to the CPU budget.
3. **Outputs.**  Weights ``runs/explore/vision/tile_decoder_v1.pt`` (gitignored); model card
   ``continuous_brain_organs/cards/tile_decoder_v1.json`` (committed; holds the weights' SHA-256); evaluation
   report ``evidence/explore/vision/tile_decoder_v1_eval.json`` (committed).  The evaluation runs the real
   ``VisionOrgan`` (hash check included) on the held-out test seeds' full 64 x 64 observations.

``--scale`` shrinks every split (pilots); ``--reuse-data`` skips generation when the stored manifest matches.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# Sleep instead of spin-waiting in OpenMP barriers.  Measured on this 2-CPU machine while another process held one
# CPU: 2 torch threads trained at 142 samples/s with the default (spinning) policy and 827 samples/s with PASSIVE.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from continuous_brain_organs import vision as V  # noqa: E402
from continuous_brain_organs import vision_data as D  # noqa: E402
from continuous_brain_world.crafter_bridge import crafter_provenance  # noqa: E402

NAME = "tile_decoder_v1"
SEED_FAMILY = "w1-vision-eng"
# (behaviour, episodes, max_steps, stride) per split; seeds are consecutive from the split's base, one per episode.
PLAN = {
    "train": {"seed_base": 72000, "episodes": [("scripted", 36, 2000, 3), ("explorer", 50, 1200, 2),
                                               ("gardener", 16, 1500, 2), ("random", 40, 600, 1)]},
    "val": {"seed_base": 73000, "episodes": [("scripted", 4, 2000, 3), ("explorer", 5, 1200, 2),
                                             ("gardener", 2, 1500, 2), ("random", 4, 600, 1)]},
    "test": {"seed_base": 74000, "episodes": [("scripted", 8, 2000, 3), ("explorer", 10, 1200, 2),
                                              ("gardener", 4, 1500, 2), ("random", 8, 600, 1)]},
}
TRAIN = {"width": 64, "blocks": 3, "epochs": 12, "batch": 256, "lr": 2e-3, "weight_decay": 1e-4,
         "warmup_steps": 200, "seed": 1, "threads": 2, "object_loss_weight": 1.0}
RARE_MATERIALS = ("diamond", "lava", "iron")
RARE_OBJECTS = ("skeleton", "arrow", "plant_ripe")


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------------------------------------- data
def split_tasks(split: str, scale: float) -> list[tuple]:
    plan = PLAN[split]
    tasks, seed = [], plan["seed_base"]
    for behaviour, n, max_steps, stride in plan["episodes"]:
        k = max(1, int(round(n * scale)))
        for i in range(n):
            seed += 1
            if i < k:
                tasks.append((seed, behaviour, max_steps, stride))
    return tasks


def _run_task(task):
    torch.set_num_threads(1)
    return D.generate_episode(*task)


def generate(split: str, scale: float, workers: int) -> tuple[dict, list]:
    tasks = split_tasks(split, scale)
    if workers > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            parts = list(ex.map(_run_task, tasks))       # ordered by task: deterministic
    else:
        parts = [_run_task(t) for t in tasks]
    return D.concat(parts), tasks


def get_data(out: Path, scale: float, workers: int, reuse: bool) -> tuple[dict, dict]:
    splits, info = {}, {}
    for split in ("train", "val", "test"):
        d = out / "data_v1" / split
        tasks = split_tasks(split, scale)
        meta = {"split": split, "tasks": [list(t) for t in tasks], "seed_family": SEED_FAMILY}
        man = d / "manifest.json"
        if reuse and man.exists() and json.loads(man.read_text())["meta"] == meta:
            data = D.load_split(d)
            manifest = json.loads(man.read_text())
            seconds = manifest.get("generation_seconds")
            print(f"[data] {split}: reused {manifest['n']} frames", flush=True)
        else:
            t0 = time.perf_counter()
            data, _ = generate(split, scale, workers)
            seconds = round(time.perf_counter() - t0, 1)
            manifest = D.save_split(data, d, meta)
            manifest["generation_seconds"] = seconds
            man.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            print(f"[data] {split}: {manifest['n']} frames in {seconds} s", flush=True)
        splits[split] = data
        info[split] = summarize_split(data, tasks, manifest, seconds)
    return splits, info


def summarize_split(data, tasks, manifest, seconds) -> dict:
    seeds = sorted({t[0] for t in tasks})
    beh = {b: int((data["behaviour"] == i).sum()) for i, b in enumerate(D.BEHAVIOURS)}
    return {"frames": int(len(data["frames"])), "tiles": int(data["material"].size),
            "episodes": len(tasks), "seeds": [seeds[0], seeds[-1]] if seeds else [],
            "seed_list": seeds,
            "episodes_by_behaviour": {b: sum(1 for t in tasks if t[1] == b) for b in D.BEHAVIOURS},
            "frames_by_behaviour": beh, "night_frames": int((data["daylight"] < 0.5).sum()),
            "sleeping_frames": int(data["sleeping"].sum()), "sha256": manifest["sha256"],
            "generation_seconds": seconds,
            "material_tile_counts": dict(zip(V.MATERIALS, np.bincount(data["material"].ravel().astype(np.int64),
                                                                      minlength=len(V.MATERIALS)).tolist())),
            "object_tile_counts": dict(zip(V.OBJECTS, np.bincount(data["object"].ravel().astype(np.int64),
                                                                  minlength=len(V.OBJECTS)).tolist()))}


# ------------------------------------------------------------------------------------------------------ training
def predict(decoder, frames: np.ndarray, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    decoder.eval()
    mats, objs = [], []
    with torch.inference_mode():
        for i in range(0, len(frames), batch):
            ml, ol = decoder(V.frames_to_tensor(np.asarray(frames[i:i + batch])))
            mats.append(ml.argmax(1).numpy().astype(np.int8))
            objs.append(ol.argmax(1).numpy().astype(np.int8))
    decoder.train()
    return np.concatenate(mats), np.concatenate(objs)


def quick_metrics(pm, po, data) -> dict:
    night = data["daylight"] < 0.5
    out = {"material_acc": float((pm == data["material"]).mean()), "object_acc": float((po == data["object"]).mean())}
    if night.any():
        out["material_acc_night"] = float((pm[night] == data["material"][night]).mean())
        out["object_acc_night"] = float((po[night] == data["object"][night]).mean())
    return out


def train(data: dict, val: dict, cfg: dict) -> tuple[V.TileDecoder, dict]:
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    # Flush subnormal floats to zero.  Without it the epochs slowed down progressively on this CPU (85 s, 185 s,
    # > 300 s for the first three) as weights, gradients and Adam moments decayed into the subnormal range.
    torch.set_flush_denormal(True)
    decoder = V.TileDecoder(width=cfg["width"], blocks=cfg["blocks"])
    crops = torch.from_numpy(np.ascontiguousarray(V.world_view_crop(data["frames"])))   # [N, 49, 63, 3] uint8
    ym = torch.from_numpy(data["material"].astype(np.int64))
    yo = torch.from_numpy(data["object"].astype(np.int64))
    n = len(crops)
    steps_per_epoch = n // cfg["batch"]
    total = steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    def lr_at(s):
        if s < cfg["warmup_steps"]:
            return (s + 1) / cfg["warmup_steps"]
        p = (s - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator().manual_seed(cfg["seed"])
    history, step = [], 0
    t0 = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=gen)
        run_m = run_o = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg["batch"]:(b + 1) * cfg["batch"]]
            x = crops[idx].permute(0, 3, 1, 2).float()
            ml, ol = decoder(x)
            lm = F.cross_entropy(ml, ym[idx])
            lo = F.cross_entropy(ol, yo[idx])
            loss = lm + cfg["object_loss_weight"] * lo
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_m += lm.item()
            run_o += lo.item()
        pm, po = predict(decoder, val["frames"])
        rec = {"epoch": epoch + 1, "step": step, "seconds": round(time.perf_counter() - t0, 1),
               "train_loss_material": round(run_m / steps_per_epoch, 5),
               "train_loss_object": round(run_o / steps_per_epoch, 5),
               "val": {k: round(v, 5) for k, v in quick_metrics(pm, po, val).items()}}
        history.append(rec)
        print(f"[train] {json.dumps(rec)}", flush=True)
    wall = time.perf_counter() - t0
    return decoder, {"history": history, "wall_seconds": round(wall, 1), "steps": step,
                     "steps_per_epoch": steps_per_epoch, "samples_seen": step * cfg["batch"]}


# ---------------------------------------------------------------------------------------------------- evaluation
def prf(conf: np.ndarray, names) -> dict:
    out = {}
    for i, name in enumerate(names):
        tp = int(conf[i, i])
        support = int(conf[i].sum())
        predicted = int(conf[:, i].sum())
        p = tp / predicted if predicted else None
        r = tp / support if support else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if support and predicted else None)
        out[name] = {"support": support, "predicted": predicted, "correct": tp,
                     "precision": None if p is None else round(p, 5), "recall": None if r is None else round(r, 5),
                     "f1": None if f1 is None else round(f1, 5)}
    return out


def confusion(y, p, k) -> np.ndarray:
    return np.bincount(y.astype(np.int64).ravel() * k + p.astype(np.int64).ravel(), minlength=k * k).reshape(k, k)


def acc(y, p, mask=None) -> dict:
    if mask is not None:
        y, p = y[mask], p[mask]
    n = int(y.size)
    return {"accuracy": round(float((y == p).mean()), 5) if n else None, "tiles": n}


def evaluate(organ: V.VisionOrgan, test: dict) -> dict:
    frames = test["frames"]
    pm, po, cm, co = [], [], [], []
    for i in range(0, len(frames), 512):
        out = organ.perceive_batch(np.asarray(frames[i:i + 512]))
        pm.append(out["material"]); po.append(out["object"])
        cm.append(out["material_conf"]); co.append(out["object_conf"])
    pm, po, cm, co = map(np.concatenate, (pm, po, cm, co))
    ym, yo = test["material"], test["object"]
    n = len(frames)
    night = np.broadcast_to((test["daylight"] < 0.5)[:, None, None], ym.shape)
    sleeping = np.broadcast_to(test["sleeping"][:, None, None], ym.shape)
    player = np.zeros(ym.shape, bool)
    player[:, V.PLAYER_ROW, V.PLAYER_COL] = True
    perceivable = test["perceivable"].astype(bool)
    edge = np.zeros(ym.shape, bool)
    edge[:, [0, -1], :] = True
    edge[:, :, [0, -1]] = True
    conf_m = confusion(ym, pm, len(V.MATERIALS))
    conf_o = confusion(yo, po, len(V.OBJECTS))
    both = (pm == ym) & (po == yo)

    def split_report(mask):
        return {"material": acc(ym, pm, mask), "object": acc(yo, po, mask),
                "both": {"accuracy": round(float(both[mask].mean()), 5) if mask.any() else None,
                         "tiles": int(mask.sum())}}

    behaviour = {}
    for i, b in enumerate(D.BEHAVIOURS):
        m = np.broadcast_to((test["behaviour"] == i)[:, None, None], ym.shape)
        if m.any():
            behaviour[b] = split_report(m)
    pm_cls = prf(conf_m, V.MATERIALS)
    po_cls = prf(conf_o, V.OBJECTS)
    # calibration-ish: accuracy by confidence band
    bands = [0.0, 0.5, 0.9, 0.99, 1.01]
    calib = {}
    for head, conf, y, p in (("material", cm, ym, pm), ("object", co, yo, po)):
        calib[head] = []
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = (conf >= lo) & (conf < hi)
            calib[head].append({"conf_range": [lo, min(hi, 1.0)], "tiles": int(m.sum()),
                                "accuracy": round(float((y[m] == p[m]).mean()), 5) if m.any() else None})
    rare = {"material": {k: {kk: pm_cls[k][kk] for kk in ("support", "precision", "recall", "f1")}
                         for k in RARE_MATERIALS},
            "object": {k: {kk: po_cls[k][kk] for kk in ("support", "precision", "recall", "f1")}
                       for k in RARE_OBJECTS}}
    for k in RARE_MATERIALS:
        i = V.MATERIAL_INDEX[k]
        rare["material"][k]["recall_day"] = acc(ym, pm, (ym == i) & ~night)
        rare["material"][k]["recall_night"] = acc(ym, pm, (ym == i) & night)
    for k in RARE_OBJECTS:
        i = V.OBJECT_INDEX[k]
        rare["object"][k]["recall_day"] = acc(yo, po, (yo == i) & ~night)
        rare["object"][k]["recall_night"] = acc(yo, po, (yo == i) & night)
    return {
        "frames": int(n), "tiles": int(ym.size),
        "overall": split_report(np.ones(ym.shape, bool)),
        "frame_exact": {"material_all_63_correct": round(float((pm == ym).all((1, 2)).mean()), 5),
                        "object_all_63_correct": round(float((po == yo).all((1, 2)).mean()), 5)},
        "day": split_report(~night), "night": split_report(night),
        "night_perceivable": split_report(night & perceivable),
        "night_drowned": split_report(night & ~perceivable),
        "sleeping": split_report(sleeping), "awake": split_report(~sleeping),
        "player_tile": split_report(player), "other_tiles": split_report(~player),
        "edge_tiles": split_report(edge), "interior_tiles": split_report(~edge),
        "by_behaviour": behaviour,
        "per_class": {"material": pm_cls, "object": po_cls},
        "per_class_night_recall": {
            "material": {k: acc(ym, pm, (ym == i) & night) for i, k in enumerate(V.MATERIALS)},
            "object": {k: acc(yo, po, (yo == i) & night) for i, k in enumerate(V.OBJECTS)}},
        "macro_f1": {"material": round(float(np.mean([v["f1"] for v in pm_cls.values() if v["f1"] is not None])), 5),
                     "object": round(float(np.mean([v["f1"] for v in po_cls.values() if v["f1"] is not None])), 5)},
        "rare_classes": rare,
        "confidence_bands": calib,
        "confusion": {"material": {"labels": list(V.MATERIALS), "rows_true_cols_pred": conf_m.tolist()},
                      "object": {"labels": list(V.OBJECTS), "rows_true_cols_pred": conf_o.tolist()}},
    }


def speed(organ: V.VisionOrgan, frames: np.ndarray) -> dict:
    torch.set_num_threads(1)
    frames = np.asarray(frames[:2000])
    for f in frames[:50]:
        organ.perceive(f)
    t = time.perf_counter()
    for f in frames:
        organ.perceive(f)
    single = len(frames) / (time.perf_counter() - t)
    t = time.perf_counter()
    for i in range(0, len(frames), 256):
        organ.perceive_batch(frames[i:i + 256])
    batched = len(frames) / (time.perf_counter() - t)
    return {"threads": 1, "frames": int(len(frames)), "single_frame_fps": round(single, 1),
            "batch256_fps": round(batched, 1),
            "note": "VisionOrgan.perceive end to end (crop, tensor, decoder, softmax, numpy) on this 2-CPU machine "
                    "with one torch thread; other processes may share the CPU"}


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs" / "explore" / "vision"))
    ap.add_argument("--card", default=str(REPO / "continuous_brain_organs" / "cards" / f"{NAME}.json"))
    ap.add_argument("--report", default=str(REPO / "evidence" / "explore" / "vision" / f"{NAME}_eval.json"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reuse-data", action="store_true")
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--width", type=int, default=TRAIN["width"])
    args = ap.parse_args()
    cfg = dict(TRAIN, epochs=args.epochs, width=args.width)
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    splits, data_info = get_data(out, args.scale, args.workers, args.reuse_data)
    data_seconds = round(time.perf_counter() - t0, 1)
    seed_sets = [set(data_info[s]["seed_list"]) for s in ("train", "val", "test")]
    assert not (seed_sets[0] & seed_sets[1] or seed_sets[0] & seed_sets[2] or seed_sets[1] & seed_sets[2])

    decoder, train_info = train(splits["train"], splits["val"], cfg)
    weights = out / f"{NAME}.pt"
    sha = V.save_weights(decoder, weights)
    print(f"[weights] {weights} sha256 {sha}", flush=True)

    card = {
        "format": "cb-organ-card/1", "name": NAME, "organ": "vision",
        "summary": "Pixels -> perceived local tile map of W1/Crafter's 9 x 7 world view (material and object per "
                   "tile). Trained in a supervised developmental phase on W1's privileged teacher labels.",
        "status": "engineering result on the exploration lane; not preregistered",
        "architecture": {"class": "continuous_brain_organs.vision.TileDecoder", "config": decoder.config,
                         "n_parameters": decoder.n_parameters(),
                         "description": "7x7 stride-7 tile-aligned patch embedding + learned per-cell position "
                                        "embedding, per-tile 2-layer MLP (1x1 convs), %d residual blocks of two 3x3 "
                                        "convs over the 7x9 tile grid, 1x1 material and object heads; GELU; input "
                                        "frame[0:49, 0:63] scaled to [-1, 1]" % decoder.config["blocks"]},
        "geometry": dict(V.GEOMETRY),
        "conventions": {"array_shape": [V.GRID_ROWS, V.GRID_COLS], "index": "[row, col]",
                        "row_0": "north (smallest world y; Crafter move_up = (0, -1))",
                        "col_0": "west (smallest world x; Crafter move_left = (-1, 0))",
                        "player_cell": [V.PLAYER_ROW, V.PLAYER_COL],
                        "world_of_cell": "(x, y) = (player_x + col - 4, player_y + row - 3)"},
        "label_space": V.label_space(),
        "training": {
            "phase": "supervised developmental phase with a teacher: labels are W1's privileged ground truth (the "
                     "material map and object list inside the visible window, labels_from_privileged); the organ "
                     "never reads them at run time",
            "seed_family": SEED_FAMILY,
            "splits": {s: {k: v for k, v in data_info[s].items() if k != "seed_list"} for s in data_info},
            "plan": PLAN, "scale": args.scale, "config": cfg, "loss": "cross-entropy (material) + cross-entropy "
            "(object), unweighted", "optimizer": "AdamW, linear warm-up then cosine decay to 0",
            "deterministic": "torch.manual_seed + torch.use_deterministic_algorithms(True), fixed epoch count, "
                             "seeded permutation; data generation is seeded per episode",
            "data_generation_seconds": round(sum(data_info[s]["generation_seconds"] or 0 for s in data_info), 1),
            "data_stage_seconds_this_run": data_seconds, "data_workers": args.workers,
            "training_wall_seconds": train_info["wall_seconds"],
            "steps": train_info["steps"], "samples_seen": train_info["samples_seen"],
            "history": train_info["history"],
        },
        "weights": {"file": str(weights.relative_to(REPO)) if weights.is_relative_to(REPO) else str(weights),
                    "sha256": sha, "bytes": weights.stat().st_size, "format": "torch.save(state_dict)"},
        "provenance": {"git_commit": git("rev-parse", "HEAD"),
                       "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                       "crafter": crafter_provenance(), "torch": torch.__version__, "numpy": np.__version__,
                       "python": platform.python_version(), "machine": platform.machine(),
                       "cpus": os.cpu_count(), "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
                       "flush_denormal": True},
    }
    Path(args.card).parent.mkdir(parents=True, exist_ok=True)
    Path(args.card).write_text(json.dumps(card, indent=1))

    organ = V.VisionOrgan(weights, args.card, threads=2)          # the real organ: verifies the hash
    t1 = time.perf_counter()
    report = evaluate(organ, splits["test"])
    report["inference_speed"] = speed(organ, splits["test"]["frames"])
    eval_seconds = round(time.perf_counter() - t1, 1)
    report = {"format": "cb-organ-eval/1", "name": NAME, "weights_sha256": sha,
              "test_split": {k: v for k, v in data_info["test"].items() if k != "seed_list"},
              "test_seeds": data_info["test"]["seed_list"],
              "definitions": {
                  "night": "daylight < 0.5 (Crafter's noise rule; W1 'dark')",
                  "night_perceivable / night_drowned": "night tiles W1's perception.perceivable_grid counts as "
                                                       "perceivable (noise weight < 0.5 at the tile centre) / not",
                  "player_tile": f"cell [{V.PLAYER_ROW}, {V.PLAYER_COL}]",
                  "edge_tiles": "rows 0 and 6, cols 0 and 8",
                  "both": "material and object both correct on the tile",
                  "f1": "null when the class has no support and no predictions"},
              "eval_seconds": eval_seconds, **report}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1))

    o = report["overall"]
    card["evaluation"] = {
        "report": str(Path(args.report).relative_to(REPO)) if Path(args.report).is_relative_to(REPO) else args.report,
        "test_frames": report["frames"], "test_tiles": report["tiles"],
        "material_accuracy": o["material"]["accuracy"], "object_accuracy": o["object"]["accuracy"],
        "both_accuracy": o["both"]["accuracy"],
        "day": {k: report["day"][k]["accuracy"] for k in ("material", "object")},
        "night": {k: report["night"][k]["accuracy"] for k in ("material", "object")},
        "night_perceivable": {k: report["night_perceivable"][k]["accuracy"] for k in ("material", "object")},
        "night_drowned": {k: report["night_drowned"][k]["accuracy"] for k in ("material", "object")},
        "player_tile": {k: report["player_tile"][k]["accuracy"] for k in ("material", "object")},
        "macro_f1": report["macro_f1"],
        "rare_classes": {h: {k: {kk: v[kk] for kk in ("support", "recall", "precision", "f1")}
                             for k, v in report["rare_classes"][h].items()} for h in ("material", "object")},
        "inference_speed": report["inference_speed"],
    }
    card["wall_seconds_total"] = round(time.time() - started, 1)
    Path(args.card).write_text(json.dumps(card, indent=1))
    V.VisionOrgan(weights, args.card)                              # the final card still verifies
    print(json.dumps(card["evaluation"], indent=1))


if __name__ == "__main__":
    main()
