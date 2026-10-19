"""Training loop and evaluation for the shared-selection proposal on
CuedTMazeWorld, comparing against the real DevelopmentBrain+ExperienceMemory
baseline and an independent-selection control, with interventions
separating memory retention, recurrent-state retention, and coordination.

NOT executed against real PyTorch in this sandbox -- reviewed against the
actual DevelopmentBrain/ExperienceMemory interfaces (lifelong/model.py,
lifelong/memory.py) and against competitive_development_brain.py, which
has its own numpy-verified zero-init claims. Hand-specified-controller
results ARE executed, in plain Python, in this same sandbox, and reported
in RESULTS section at the bottom of PROPOSAL.md, not here -- this file is
the torch-dependent half.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .competitive_development_brain import CompetitiveDevelopmentBrain
from .cued_tmaze_world import CuedTMazeWorld
from .model import DevelopmentBrain, BrainConfig, load_model, model_payload
from .memory import ExperienceMemory

CORRIDOR_LENGTHS = (1, 2, 3, 4, 5)  # short-to-long, specifically to separate GRU-only retention from external memory
PASS_THRESHOLDS = {"min_correct_rate": 0.80, "advantage_over_independent": 0.10}


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bootstrap_ci(diffs: np.ndarray, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def one_hot5(action, device):
    return F.one_hot(torch.tensor([action], device=device), 5).float()


def run_episode(model, kind, seed, corridor_length, *, use_memory: bool, use_competition: bool,
                reset_hidden_every: int | None, device: str):
    """
    use_memory=False: ExperienceMemory.retrieve() is never called; the
    memory input is fixed at zero every step -- isolates whether the
    EXTERNAL buffer is doing anything, independent of the GRU's own state.
    reset_hidden_every=N: hidden state is forcibly zeroed every N steps --
    isolates whether the RECURRENT state alone is what's carrying the cue,
    independent of external memory. Both can be varied independently;
    neither is coupled to `kind` (baseline/shared/independent), so all
    combinations are meaningful.
    """
    env = CuedTMazeWorld(seed=seed, corridor_length=corridor_length)
    obs = env.reset()
    memory = ExperienceMemory()
    hidden = model.initial(1, device)
    style = np.zeros(4, dtype=np.float32)  # STYLE_DIM=4; unused by this task, kept for interface compatibility
    step = 0
    while not env.done:
        step += 1
        if reset_hidden_every and step % reset_hidden_every == 0:
            hidden = model.initial(1, device)
        retrieved = memory.retrieve(obs["sensors"]) if use_memory else np.zeros(8, np.float32)  # MEMORY_DIM=8
        grid_t = torch.as_tensor(obs["grid"][None], dtype=torch.float32, device=device)
        sensors_t = torch.as_tensor(obs["sensors"][None], dtype=torch.float32, device=device)
        memory_t = torch.as_tensor(retrieved[None], dtype=torch.float32, device=device)
        style_t = torch.as_tensor(style[None], dtype=torch.float32, device=device)

        if isinstance(model, CompetitiveDevelopmentBrain):
            out = model(grid_t, sensors_t, memory_t, style_t, hidden, use_competition=use_competition)
        else:
            out = model(grid_t, sensors_t, memory_t, style_t, hidden)
        hidden = out["hidden"]
        action = int(torch.argmax(model.choose(out) if hasattr(model, "choose") else out["logits"], dim=-1)[0].item())

        obs2, reward, done, info = env.step(action)
        if use_memory:
            memory.add(obs["sensors"], action, reward, {"resource": None, "collision": False})
        obs = obs2
    return {"seed": seed, "corridor_length": corridor_length, "correct": bool(info["reached_correct"])}


def _summary(rows):
    return {"episodes": len(rows), "correct_rate": float(statistics.fmean(float(r["correct"]) for r in rows))}


def evaluate(checkpoint_path: str, *, episodes_per_length=100, seed_start=81001, device="cpu",
             output: str | None = None):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model = load_model(payload, device)
    shared_model = CompetitiveDevelopmentBrain.grow_from(base_model, use_shared=True).to(device).eval()
    independent_model = CompetitiveDevelopmentBrain.grow_from(base_model, use_shared=False).to(device).eval()
    base_model.eval()

    conditions = {
        "baseline_as_is": (base_model, dict(use_memory=True, use_competition=False, reset_hidden_every=None)),
        "shared_selection": (shared_model, dict(use_memory=True, use_competition=True, reset_hidden_every=None)),
        "independent_selection": (independent_model, dict(use_memory=True, use_competition=True, reset_hidden_every=None)),
        # Interventions, run against the shared-selection model specifically:
        "shared_no_external_memory": (shared_model, dict(use_memory=False, use_competition=True, reset_hidden_every=None)),
        "shared_hidden_reset_every_1": (shared_model, dict(use_memory=True, use_competition=True, reset_hidden_every=1)),
        "shared_competition_disabled": (shared_model, dict(use_memory=True, use_competition=False, reset_hidden_every=None)),
    }

    results = {}
    for name, (model, kwargs) in conditions.items():
        by_length = {}
        for L in CORRIDOR_LENGTHS:
            rows = [run_episode(model, "cued_tmaze", seed_start + i, L, device=device, **kwargs)
                    for i in range(episodes_per_length)]
            by_length[L] = _summary(rows)
        results[name] = by_length
        print(f"{name}: " + ", ".join(f"L={L}:{by_length[L]['correct_rate']:.2f}" for L in CORRIDOR_LENGTHS))

    diffs_vs_independent = {
        L: np.array([1.0 if run_episode(shared_model, "cued_tmaze", seed_start + i, L, device=device,
                                        use_memory=True, use_competition=True, reset_hidden_every=None)["correct"] else 0.0
                     for i in range(episodes_per_length)])
        - np.array([1.0 if run_episode(independent_model, "cued_tmaze", seed_start + i, L, device=device,
                                       use_memory=True, use_competition=True, reset_hidden_every=None)["correct"] else 0.0
                    for i in range(episodes_per_length)])
        for L in CORRIDOR_LENGTHS
    }
    ci_by_length = {L: _bootstrap_ci(d) for L, d in diffs_vs_independent.items()}

    out = {
        "format": "continuous_brain.competitive_gating_gate.v1",
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "corridor_lengths": list(CORRIDOR_LENGTHS),
        "episodes_per_length": episodes_per_length,
        "results": results,
        "shared_minus_independent_paired_ci": {str(L): {"mean": float(d.mean()), "ci_lo": ci[0], "ci_hi": ci[1]}
                                               for (L, d), ci in zip(diffs_vs_independent.items(), ci_by_length.values())},
        "pass_thresholds_fixed_before_evaluation": PASS_THRESHOLDS,
        "reporting_note": "Compare 'shared_selection' against 'independent_selection', not against "
                          "'baseline_as_is', for the coordination claim specifically. 'shared_no_external_memory' "
                          "isolates whether the external buffer matters; 'shared_hidden_reset_every_1' isolates "
                          "whether the GRU's own recurrence matters (a hidden reset every step removes recurrent "
                          "carry-over entirely, leaving only whatever the external memory buffer supplies); "
                          "'shared_competition_disabled' should reproduce 'baseline_as_is' within noise -- if it "
                          "doesn't, the zero-init growth claim in competitive_development_brain.py has failed in "
                          "a way its numpy verification did not catch, and that discrepancy is itself the finding "
                          "to report.",
    }
    if output:
        Path(output).write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def train(checkpoint_path: str, *, updates: int = 4000, seed: int = 81501, learning_rate: float = 1e-4,
          use_shared: bool = True, corridor_lengths=CORRIDOR_LENGTHS, device: str = "cpu",
          state_dir: str = "state_competitive_gating"):
    """
    Trains ONLY the parameters returned by model.trainable_parameters()
    -- the competition module plus the four residual heads. base.* is
    frozen (requires_grad_(False) in __init__) and receives no optimizer
    updates regardless of what this loop does; gradient passes through it
    to reach the residuals and competition module, per standard adapter
    training, without ever touching its own parameters.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model = load_model(payload, device)
    model = CompetitiveDevelopmentBrain.grow_from(base_model, use_shared=use_shared).to(device)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    for update in range(int(updates)):
        L = int(rng.choice(corridor_lengths))
        episode_seed = int(rng.integers(0, 10_000_000))
        env = CuedTMazeWorld(seed=episode_seed, corridor_length=L)
        obs = env.reset()
        memory = ExperienceMemory()
        hidden = model.initial(1, device)
        style = torch.zeros(1, 4, device=device)
        log_probs, total_reward = [], 0.0
        while not env.done:
            retrieved = memory.retrieve(obs["sensors"])
            grid_t = torch.as_tensor(obs["grid"][None], dtype=torch.float32, device=device)
            sensors_t = torch.as_tensor(obs["sensors"][None], dtype=torch.float32, device=device)
            memory_t = torch.as_tensor(retrieved[None], dtype=torch.float32, device=device)
            out = model(grid_t, sensors_t, memory_t, style, hidden, use_competition=True)
            hidden = out["hidden"]
            dist = torch.distributions.Categorical(logits=model.choose(out))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs2, reward, done, info = env.step(int(action.item()))
            memory.add(obs["sensors"], int(action.item()), reward,
                       {"resource": None, "collision": False})
            total_reward += reward
            obs = obs2
        # Plain REINFORCE on episode return -- sufficient for this small,
        # short-horizon task and consistent with keeping this delivery's
        # scope to "does the mechanism work at all," not a tuned RL recipe.
        loss = -torch.stack(log_probs).sum() * total_reward
        opt.zero_grad(); loss.backward(); opt.step()
        if (update + 1) % 200 == 0:
            print(f"train[{'shared' if use_shared else 'independent'}] update={update+1}/{updates} "
                  f"last_return={total_reward:.3f}")

    Path(state_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(state_dir) / f"competitive_{'shared' if use_shared else 'independent'}.pt"
    torch.save({"config": vars(model.cfg) if hasattr(model.cfg, "__dict__") else model.cfg.__dict__,
                "model": model.state_dict(), "use_shared": use_shared,
                "base_checkpoint_sha256": _sha256_file(checkpoint_path)}, out_path)
    return {"output": str(out_path)}


def load_trained(path: str, base_checkpoint: str, device: str = "cpu") -> CompetitiveDevelopmentBrain:
    saved = torch.load(path, map_location=device, weights_only=False)
    base_payload = torch.load(base_checkpoint, map_location=device, weights_only=False)
    base_model = load_model(base_payload, device)
    model = CompetitiveDevelopmentBrain.grow_from(base_model, use_shared=saved["use_shared"]).to(device)
    model.load_state_dict(saved["model"])
    return model
