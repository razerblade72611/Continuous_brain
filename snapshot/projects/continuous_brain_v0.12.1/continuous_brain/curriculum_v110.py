from __future__ import annotations

"""v0.11.0 generalized embodied curriculum.

These environments preserve the child's 5-action body interface while varying
barrier geometry between episodes.  Collection uses a geometry-aware *data
collection policy* only to create successful lived trajectories; motor_valid is
always false and qualification is child-only.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch

from .environment import DetourWorld
from .parented_data import NurseryDataset
from .parented_train import load_checkpoint
from .planning_v100 import _env_context, _env_image_tensor
from .planning_v103 import _hierarchical_action
from .world_consolidation_v104 import transition_digest


@dataclass
class RandomizedDetourWorld(DetourWorld):
    """Single barrier, but wall/gap/spawn geometry changes every episode."""
    layout_seed_offset: int = 991

    def reset(self) -> np.ndarray:
        # __post_init__ in the base class has already created self.rng.
        self.wall_x = self.rng.uniform(-0.18, 0.18)
        self.gap_center_y = self.rng.uniform(-0.62, 0.62)
        self.gap_half_height = self.rng.uniform(0.13, 0.20)
        left_to_right = self.rng.random() < 0.5
        if left_to_right:
            self.agent[:] = [self.rng.uniform(-0.88, self.wall_x - 0.36), self.rng.uniform(-0.55, 0.35)]
            self.target[:] = [self.rng.uniform(self.wall_x + 0.36, 0.88), self.rng.uniform(-0.45, 0.45)]
        else:
            self.agent[:] = [self.rng.uniform(self.wall_x + 0.36, 0.88), self.rng.uniform(-0.55, 0.35)]
            self.target[:] = [self.rng.uniform(-0.88, self.wall_x - 0.36), self.rng.uniform(-0.45, 0.45)]
        self.energy = 1.0; self.last_reward = 0.0; self.steps = 0; self.collisions = 0
        return self._obs()

    def visual_snapshot(self, predicted_obs=None) -> dict:
        return {
            "environment_type": "randomized_detour",
            "wall_x": float(self.wall_x),
            "gap_center_y": float(self.gap_center_y),
            "gap_half_height": float(self.gap_half_height),
        }


@dataclass
class TwoBarrierWorld(DetourWorld):
    """Two vertical barriers with independently randomized openings."""
    wall_x2: float = 0.38
    gap_center_y2: float = -0.45
    gap_half_height2: float = 0.16

    def reset(self) -> np.ndarray:
        self.wall_x = self.rng.uniform(-0.48, -0.16)
        self.wall_x2 = self.rng.uniform(0.16, 0.48)
        self.gap_center_y = self.rng.uniform(0.30, 0.68)
        self.gap_center_y2 = self.rng.uniform(-0.68, -0.30)
        self.gap_half_height = self.rng.uniform(0.13, 0.19)
        self.gap_half_height2 = self.rng.uniform(0.13, 0.19)
        self.agent[:] = [self.rng.uniform(-0.92, self.wall_x - 0.22), self.rng.uniform(-0.35, 0.25)]
        self.target[:] = [self.rng.uniform(self.wall_x2 + 0.22, 0.92), self.rng.uniform(-0.25, 0.35)]
        self.energy = 1.0; self.last_reward = 0.0; self.steps = 0; self.collisions = 0
        return self._obs()

    def _in_gap2(self, y: float) -> bool:
        return abs(float(y) - float(self.gap_center_y2)) <= float(self.gap_half_height2)

    def _blocked(self, old: np.ndarray, proposed: np.ndarray) -> bool:
        for wx, fn in ((self.wall_x, self._in_gap), (self.wall_x2, self._in_gap2)):
            lo, hi = min(float(old[0]), float(proposed[0])), max(float(old[0]), float(proposed[0]))
            if lo <= wx + self.wall_half_width and hi >= wx - self.wall_half_width:
                if not fn(0.5 * (float(old[1]) + float(proposed[1]))):
                    return True
        return False

    def render_rgb(self, size: int = 96) -> np.ndarray:
        # Start with base rendering for agent/target/grid and redraw both barriers.
        img = super().render_rgb(size)
        size = int(size)
        def pix(pos):
            x = int(round((float(pos[0]) + 1.0) * 0.5 * (size - 1)))
            y = int(round((1.0 - (float(pos[1]) + 1.0) * 0.5) * (size - 1)))
            return x, y
        thickness = max(2, size // 32)
        for wx, gy, gh in ((self.wall_x, self.gap_center_y, self.gap_half_height),
                           (self.wall_x2, self.gap_center_y2, self.gap_half_height2)):
            wall_px, _ = pix((wx,0.0)); _, a = pix((0.0,gy+gh)); _, b = pix((0.0,gy-gh)); lo,hi=sorted((a,b))
            img[:, max(0,wall_px-thickness):min(size,wall_px+thickness+1)] = np.array([185,70,70],dtype=np.uint8)
            img[lo:hi+1, max(0,wall_px-thickness):min(size,wall_px+thickness+1)] = np.array([35,80,45],dtype=np.uint8)
        return img

    def visual_snapshot(self, predicted_obs=None) -> dict:
        return {
            "environment_type": "two_barrier",
            "walls": [float(self.wall_x), float(self.wall_x2)],
            "gaps": [float(self.gap_center_y), float(self.gap_center_y2)],
        }


def _save_rgb(path: Path, array: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(array).save(path)


def _context(obs: np.ndarray) -> np.ndarray:
    out=np.zeros(48,dtype=np.float32); out[:6]=obs[:6]; out[6]=float(obs[4]); out[30]=1.0; return out


def _toward(delta: float, positive_action: int, negative_action: int) -> int:
    return positive_action if delta > 0 else negative_action


def _collector_action(env, obs: np.ndarray, rng: random.Random, exploration: float) -> int:
    if rng.random() < float(exploration): return rng.randrange(5)
    x,y=float(env.agent[0]),float(env.agent[1]); tx,ty=float(env.target[0]),float(env.target[1])
    waypoints=[]
    if isinstance(env, TwoBarrierWorld):
        if x < env.wall_x + env.wall_half_width:
            waypoints=[(env.wall_x + 0.10, env.gap_center_y)]
        elif x < env.wall_x2 + env.wall_half_width:
            waypoints=[(env.wall_x2 + 0.10, env.gap_center_y2)]
    else:
        if x < env.wall_x and tx > env.wall_x:
            waypoints=[(env.wall_x + 0.10, env.gap_center_y)]
        elif x > env.wall_x and tx < env.wall_x:
            waypoints=[(env.wall_x - 0.10, env.gap_center_y)]
    wx,wy = waypoints[0] if waypoints else (tx,ty)
    dx,dy=wx-x,wy-y
    # Reach opening height first near a barrier, otherwise favor dominant displacement.
    if waypoints and abs(dy) > 0.08: return _toward(dy,1,2)
    if abs(dx) >= abs(dy): return _toward(dx,4,3)
    return _toward(dy,1,2)


def _make_env(kind: str, seed: int, max_steps: int):
    kind=str(kind).lower()
    if kind=="randomized_detour": return RandomizedDetourWorld(seed=int(seed),max_episode_steps=int(max_steps))
    if kind=="two_barrier": return TwoBarrierWorld(seed=int(seed),max_episode_steps=int(max_steps))
    raise ValueError("kind must be randomized_detour or two_barrier")


def collect_generalized_curriculum(directory: str | Path, *, kind: str="randomized_detour",
                                   samples: int=8000, seed: int=31001,
                                   exploration: float=.25, max_steps: int=160) -> dict:
    directory=Path(directory); ds=NurseryDataset(directory)
    if ds.records(): raise FileExistsError("generalized curriculum destination must be empty")
    env=_make_env(kind,seed,max_steps); rng=random.Random(int(seed)); obs=env.reset(); prev=-1; eid=0; estep=0; successes=0
    for i in range(int(samples)):
        rid=f"v110-{kind}-{i:07d}"; ip=ds.images/f"{rid}.png"; npth=ds.images/f"{rid}-next.png"
        _save_rgb(ip,env.render_rgb(96)); action=_collector_action(env,obs,rng,exploration)
        nxt,reward,done,info=env.step(action); _save_rgb(npth,env.render_rgb(96))
        ds.append({"record_id":rid,"episode_id":eid,"episode_step":estep,"previous_action":prev,
                   "image":str(ip.relative_to(directory)),"next_image":str(npth.relative_to(directory)),
                   "context":_context(obs).tolist(),"next_context":_context(nxt).tolist(),
                   "executed_action":int(action),"action":int(action),"motor_valid":False,"world_valid":True,
                   "reward":float(reward),"success":float(bool(info.get("reached"))),"collision":bool(info.get("collision",False)),
                   "provenance":"lived_generalized_curriculum","source":f"v110-{kind}","layout":env.visual_snapshot()})
        if info.get("reached"): successes += 1
        if done:
            obs=env.reset(); prev=-1; eid+=1; estep=0
        else:
            obs=nxt; prev=int(action); estep+=1
        if (i+1)%1000==0: print(f"v110_curriculum={i+1}/{samples}")
    manifest={"format":"continuous_brain.generalized_curriculum.v110","kind":kind,"records":int(samples),"seed":int(seed),
              "exploration":float(exploration),"parent_models_used":False,"motor_supervision":False,
              "successful_transitions":int(successes),"transition_digest":transition_digest(directory)}
    ds.write_manifest(manifest); return manifest


def _mean(xs):
    xs=list(xs); return float(statistics.fmean(xs)) if xs else 0.0


@torch.no_grad()
def evaluate_generalized_planning(checkpoint: str | Path, *, kind: str="randomized_detour",
                                  episodes: int=100, steps: int=160, seed_start: int=41001,
                                  option_depth: int=2, selector_topk: int=2, beam_width: int=8,
                                  max_option_steps: int=12, discount: float=.95, min_commit: int=3,
                                  terminate_threshold: float=.5, device: str="cpu",
                                  output: str | Path | None=None) -> dict:
    model,_=load_checkpoint(checkpoint,device); model.eval(); rows=[]
    for j in range(int(episodes)):
        seed=int(seed_start)+j; env=_make_env(kind,seed,steps); obs=env.reset(); hidden=model.nucleus.initial_state(1,device)
        memory=torch.zeros(1,model.cfg.memory_dim,device=device); prev=torch.zeros(1,dtype=torch.long,device=device)
        current=None; age=0; completed=False; reward_total=0.0; switches=0; decisions=0
        for step in range(1,int(steps)+1):
            image=_env_image_tensor(env,model.cfg.image_size,device); context,context7=_env_context(obs,device)
            sem=model.vision(image); spa=model.spatial(image); fused=model.fuse_vision(sem,spa)
            nuc=model.nucleus(fused,context,memory,prev,hidden); hidden=nuc["hidden"]; stable=model.target_spatial_bus(image)
            dec=_hierarchical_action(model,stable,context7,current,age,option_depth=option_depth,selector_topk=selector_topk,
                                     beam_width=beam_width,max_option_steps=max_option_steps,discount=discount,
                                     min_commit=min_commit,terminate_threshold=terminate_threshold)
            current=int(dec["option"]); switches+=int(dec["switched"]); decisions+=int(dec["high_level_decision"]); age=int(dec["option_age"])+1
            action=int(dec["action"]); obs,r,done,info=env.step(action); reward_total+=float(r); prev=torch.tensor([action+1],device=device)
            if info.get("reached"): completed=True; used=step
            if done: break
        rows.append({"seed":seed,"completed":completed,"reward":reward_total,"collisions":int(env.collisions),"steps":used if completed else int(steps),"option_switches":switches,"high_level_decisions":decisions,"layout":env.visual_snapshot()})
    result={"format":"continuous_brain.generalized_planning_eval.v110","kind":kind,"episodes":int(episodes),"seed_start":int(seed_start),
            "completion_rate":_mean(float(r["completed"]) for r in rows),"mean_reward":_mean(r["reward"] for r in rows),
            "mean_collisions":_mean(r["collisions"] for r in rows),"mean_option_switches":_mean(r["option_switches"] for r in rows),
            "training_performed_during_evaluation":False,"parents_used":False,"rows":rows}
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(result,indent=2,sort_keys=True))
    return result
