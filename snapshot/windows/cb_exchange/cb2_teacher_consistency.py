"""Teacher consistency (explore lane, ENGINEERING_ONLY, CPU only).

The counterfactual labels in CB2 evaluation lives come from a ShadowRoutine: the routine library run beside the
actor, replanning from the actor's own beliefs every step.  How meaningful is "agreement with the routine"?  This
replays the ROUTINE's own lives (it acts; nothing is trained, nothing is archived) and asks, at every step, whether
a ShadowRoutine reading the very same beliefs would have chosen the action the routine actually took.  That rate
is the ceiling for any student measured against shadow labels.

Run from the explore checkout root with its venv:  python <this file> --seeds 75001-75040 --out <json>
Also checks that observing does not change the life: each routine life must match the r2 evaluation's length.
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/explore"))
OPT = ("routine", "explore", "sleeping", "eat")
# routine life lengths on 75001-75040 from the r2 evaluation (r2b_resume.log), to prove the replay is the same life
R2_ROUTINE_STEPS = [92, 681, 144, 501, 146, 588, 374, 1160, 249, 366, 282, 163, 375, 644, 218, 494, 388, 277, 1203,
                    1233, 700, 312, 785, 224, 334, 701, 408, 428, 836, 341, 453, 698, 153, 296, 613, 426, 289, 230, 714, 1508]


def one(args):
    seed, steps = args
    import torch
    torch.set_num_threads(1)
    from cb2_collect import vision
    from continuous_brain_cb2.record import ShadowRoutine, routine_choice
    from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    from continuous_brain_individual.skills import ACTIONS
    ind = Individual(IndividualConfig(), vision(), LanguageOrgan.from_config({"backend": "dev"}))
    env = W1Env(W1Config())
    obs = env.reset(seed)
    info = first_info(env)
    shadow = ShadowRoutine(seed)
    rows, done, n = [], False, 0
    while not done and n < steps:
        action, utter = ind.decide(obs, info)
        ra, ro = routine_choice(ind)
        sa, so = shadow(ind)
        rows.append((int(obs["step"]), int(action), ra, ro, sa, so))
        obs, _, done, info = env.step(action, utter)
        n += 1
    return seed, rows, list(ACTIONS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="75001-75040")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    t = time.time()
    with Pool(a.workers) as p:
        res = sorted(p.map(one, [(s, a.steps) for s in seeds]))
    names = res[0][2]
    R = np.array([r for _, rows, _ in res for r in rows])          # step, actor, r_act, r_opt, s_act, s_opt
    step, actor, ra, ro, sa, so = R.T
    agree = sa == ra
    lengths = [len(rows) for _, rows, _ in res]
    out = {"format": "cb2-teacher-consistency/1", "status": "ENGINEERING_ONLY", "seeds": a.seeds, "steps": int(len(R)),
           "seconds": round(time.time() - t, 1),
           "actor_is_routine_choice": round(float((actor == ra).mean()), 4),
           "replay_matches_r2_routine_lengths": (lengths == R2_ROUTINE_STEPS) if a.seeds == "75001-75040" else None,
           "shadow_vs_routine_action_agreement": round(float(agree.mean()), 4),
           "shadow_vs_routine_option_agreement": round(float((so == ro).mean()), 4),
           "by_routine_option": {OPT[o]: [round(float(agree[ro == o].mean()), 4), int((ro == o).sum())] for o in range(4) if (ro == o).sum()},
           "by_routine_action": {names[k]: [round(float(agree[ra == k].mean()), 4), int((ra == k).sum())] for k in range(len(names)) if (ra == k).sum() >= 30},
           "by_step_in_life": {f"{x}-{y}": [round(float(agree[(step >= x) & (step <= y)].mean()), 4), int(((step >= x) & (step <= y)).sum())]
                               for x, y in [(0, 49), (50, 99), (100, 199), (200, 399), (400, 10**6)] if ((step >= x) & (step <= y)).sum()},
           "life_lengths": lengths}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "life_lengths"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
