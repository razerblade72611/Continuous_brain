"""Train and evaluate the function-preserving competitive-broadcast branch.

The experiment uses the *real* InfantBrain checkpoint as a frozen base.  It does
not train a replacement brain and it does not mutate the source checkpoint.

Primary comparison
------------------
``shared`` uses one stream selector for recurrent state, motor policy, and world
model. ``independent`` gives those three consumers separate selectors.  A
``uniform`` control keeps the shared selector query at zero so its weights remain
exactly 1/4 while the residual adapters can still learn.

Evaluation also includes:
* the exact frozen base;
* shared selection with competition disabled (must reproduce the base);
* shared selection with hidden state reset every step (retention ablation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .competitive_infant_brain import CompetitiveInfantBrain
from .cued_detour_world import CuedDetourWorld
from .infant import InfantBrain, InfantConfig

CORRIDOR_LENGTHS = (0.3, 0.6, 1.0, 1.4)
FORMAT = "continuous_brain.competitive_broadcast.v2"


def _one_hot_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Scatter-free one-hot encoding compatible with DirectML/PrivateUse1."""
    idx = action.long().clamp(0, int(action_dim) - 1)
    classes = torch.arange(
        int(action_dim),
        device=idx.device,
        dtype=idx.dtype,
    )
    return (idx.unsqueeze(-1) == classes).to(dtype=torch.float32)


def _resolve_device(name: str | Any):
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    if key in {"directml", "dml"}:
        try:
            import torch_directml
        except ImportError as exc:  # pragma: no cover - host specific
            raise RuntimeError("DirectML requested but torch-directml is not installed") from exc
        return torch_directml.device()
    return torch.device(name)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000, seed: int = 0):
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = diffs.size
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        means[i] = diffs[rng.integers(0, n, n)].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _image_tensor(env: CuedDetourWorld, size: int, device) -> torch.Tensor:
    arr = env.render_rgb(size).astype(np.float32) / 255.0
    return torch.tensor(np.transpose(arr, (2, 0, 1))[None], dtype=torch.float32, device=device)


def _context_tensor(obs: np.ndarray, device, cfg: InfantConfig):
    full = np.zeros(int(cfg.context_dim), dtype=np.float32)
    full[:6] = obs[:6]
    if cfg.context_dim > 6:
        full[6] = obs[4]
    full_t = torch.tensor(full[None], dtype=torch.float32, device=device)
    return full_t, full_t[:, : int(cfg.dynamics_context_dim)]


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_base_checkpoint(path: str | Path) -> tuple[InfantBrain, InfantConfig, dict, str]:
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or "model" not in saved or "config" not in saved:
        raise ValueError(f"{path} is not an InfantBrain training checkpoint")
    cfg = InfantConfig(**dict(saved["config"]))
    parent_dims = {str(k): int(v) for k, v in (saved.get("vision_parent_dims") or {}).items()}
    model = InfantBrain(cfg, vision_parent_dims=parent_dims)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, cfg, parent_dims, _sha256_file(path)


def _selector_modules(model: CompetitiveInfantBrain):
    if model.use_shared:
        return [model.competition]
    return [
        model.competition.recurrent_gate,
        model.competition.motor_gate,
        model.competition.world_gate,
    ]




def _teacher_action(env: CuedDetourWorld) -> int:
    """Optimal demonstrator used only for training labels.

    The learner never receives ``env.cue`` directly after the cue window.  The
    teacher may inspect it to label the delayed choice, exactly as a parent/
    demonstrator can supervise behavior whose cause the child must remember.
    """
    # Move slightly above the junction before choosing a side so the goal's
    # vertical offset lies strictly inside the 0.15 terminal radius.
    if float(env.agent[1]) < float(env._junction_y()) + 0.08 - 1e-6:
        return 1
    target_x = -0.7 if env.cue == "left" else 0.7
    if abs(float(env.agent[0]) - target_x) <= 0.04:
        return 0
    return 3 if float(env.agent[0]) > target_x else 4


def _competition_diagnostics(comp: dict | None) -> tuple[float | None, float | None]:
    if comp is None:
        return None, None
    eps = 1e-12
    if "weights" in comp:
        w = comp["weights"]
        ent = -(w * (w.clamp_min(eps)).log()).sum(-1).mean()
        return float(ent.detach().cpu().item()), 0.0
    weights = [comp["recurrent_weights"], comp["motor_weights"], comp["world_weights"]]
    ent = torch.stack([-(w * w.clamp_min(eps).log()).sum(-1).mean() for w in weights]).mean()
    disagreement = (
        (weights[0] - weights[1]).abs().mean()
        + (weights[0] - weights[2]).abs().mean()
        + (weights[1] - weights[2]).abs().mean()
    ) / 3.0
    return float(ent.detach().cpu().item()), float(disagreement.detach().cpu().item())


def run_episode(
    model,
    seed: int,
    corridor_length: float,
    *,
    use_competition: bool,
    reset_hidden_every: int | None,
    device,
    cfg: InfantConfig,
    max_episode_steps: int = 120,
):
    env = CuedDetourWorld(
        seed=int(seed), corridor_length=float(corridor_length), max_episode_steps=int(max_episode_steps)
    )
    obs = env.reset()
    hidden = (
        model.initial_state(1, device)
        if hasattr(model, "initial_state")
        else model.nucleus.initial_state(1, device)
    )
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    world_losses, entropies, disagreements = [], [], []
    info = {"reached_correct": False}
    step_count = 0

    with torch.no_grad():
        while not env.done:
            step_count += 1
            if reset_hidden_every and step_count % int(reset_hidden_every) == 0:
                hidden = (
                    model.initial_state(1, device)
                    if hasattr(model, "initial_state")
                    else model.nucleus.initial_state(1, device)
                )

            image = _image_tensor(env, cfg.image_size, device)
            context, context_dyn = _context_tensor(obs, device, cfg)
            memory = torch.zeros(1, cfg.memory_dim, device=device)

            if isinstance(model, CompetitiveInfantBrain):
                out = model.step(
                    image, context, memory, prev_action, hidden,
                    use_competition=use_competition,
                )
                workspace_world = out["workspace_for_world"]
                comp = out["competition"]
            else:
                semantic, spatial = model.vision(image), model.spatial(image)
                fused = model.fuse_vision(semantic, spatial)
                nuc = model.nucleus(fused, context, memory, prev_action, hidden)
                out = {"hidden": nuc["hidden"], "logits": model.motor(nuc["workspace"])}
                workspace_world = nuc["workspace"]
                comp = None

            hidden = out["hidden"]
            action = torch.argmax(out["logits"], dim=-1)
            action_oh = _one_hot_action(action, cfg.action_dim)
            if isinstance(model, CompetitiveInfantBrain):
                world_out = model.predict_world(
                    workspace_world, action_oh, base_context=context_dyn
                )
            else:
                world_out = model.world(workspace_world, action_oh, base_context=context_dyn)

            obs2, _reward, _done, info = env.step(int(action.item()))
            _, next_context_dyn = _context_tensor(obs2, device, cfg)
            world_losses.append(float(F.mse_loss(world_out["next_context"], next_context_dyn).cpu().item()))
            entropy, disagreement = _competition_diagnostics(comp)
            if entropy is not None:
                entropies.append(entropy)
            if disagreement is not None:
                disagreements.append(disagreement)
            prev_action = (action + 1).long()
            obs = obs2

    return {
        "seed": int(seed),
        "corridor_length": float(corridor_length),
        "correct": bool(info["reached_correct"]),
        "steps": int(step_count),
        "world_context_mse": float(np.mean(world_losses)) if world_losses else float("nan"),
        "selection_entropy": float(np.mean(entropies)) if entropies else None,
        "selection_disagreement": float(np.mean(disagreements)) if disagreements else None,
    }


def train(
    *,
    base_checkpoint: str,
    use_shared: bool = True,
    updates: int = 4000,
    seed: int = 81501,
    learning_rate: float = 1e-4,
    freeze_competition_query: bool = False,
    corridor_lengths=CORRIDOR_LENGTHS,
    device: str | Any = "cpu",
    state_dir: str = "state_competitive_infant",
    max_episode_steps: int = 120,
    log_every: int = 200,
):
    """Train only the new pathway using parented imitation + world prediction.

    Teacher-forced trajectories deliberately remove exploration as a confound.
    The delayed side-choice label is cue-dependent after the cue has vanished,
    so learning that label still requires recurrent retention.  This mirrors the
    project's existing parented-development style better than sparse REINFORCE.
    """
    if not base_checkpoint:
        raise ValueError("base_checkpoint is required; random-base training is intentionally disallowed")
    device = _resolve_device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)

    base, cfg, parent_dims, base_sha = _load_base_checkpoint(base_checkpoint)
    model = CompetitiveInfantBrain.grow_from(base, use_shared=bool(use_shared)).to(device)
    model.train(True)

    if freeze_competition_query:
        for gate in _selector_modules(model):
            gate.query.weight.requires_grad_(False)
            gate.query.bias.requires_grad_(False)

    trainable = model.trainable_parameters()
    opt = torch.optim.Adam(trainable, lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    returns, motor_accs = [], []

    for update in range(int(updates)):
        L = float(rng.choice(tuple(corridor_lengths)))
        episode_seed = int(rng.integers(0, 2**31 - 1))
        env = CuedDetourWorld(
            seed=episode_seed, corridor_length=L, max_episode_steps=int(max_episode_steps)
        )
        obs = env.reset()
        hidden = model.initial_state(1, device)
        prev_action = torch.zeros(1, dtype=torch.long, device=device)
        motor_losses, world_losses = [], []
        total_reward, correct_actions, action_count = 0.0, 0, 0

        while not env.done:
            image = _image_tensor(env, cfg.image_size, device)
            context, context_dyn = _context_tensor(obs, device, cfg)
            memory = torch.zeros(1, cfg.memory_dim, device=device)
            out = model.step(image, context, memory, prev_action, hidden, use_competition=True)
            hidden = out["hidden"]

            teacher_action = _teacher_action(env)
            target = torch.tensor([teacher_action], dtype=torch.long, device=device)
            motor_losses.append(F.cross_entropy(out["logits"], target))
            predicted = int(torch.argmax(out["logits"], dim=-1)[0].item())
            correct_actions += int(predicted == teacher_action)
            action_count += 1

            # Execute the teacher action.  This keeps every condition on matched,
            # successful trajectories so the experiment measures representation/
            # selection learning rather than accidental exploration coverage.
            action = target
            action_oh = _one_hot_action(action, cfg.action_dim)
            world_out = model.predict_world(
                out["workspace_for_world"], action_oh, base_context=context_dyn
            )
            obs2, reward, _done, _info = env.step(teacher_action)
            _, next_context_dyn = _context_tensor(obs2, device, cfg)
            world_losses.append(F.smooth_l1_loss(world_out["next_context"], next_context_dyn))
            total_reward += float(reward)
            prev_action = (action + 1).long()
            obs = obs2

        motor_loss = torch.stack(motor_losses).mean()
        world_loss = torch.stack(world_losses).mean()
        loss = motor_loss + world_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        opt.step()
        returns.append(total_reward)
        motor_accs.append(correct_actions / max(1, action_count))

        if log_every and (update + 1) % int(log_every) == 0:
            tag = "shared" if use_shared else "independent"
            if freeze_competition_query:
                tag += "/uniform"
            window = min(len(returns), int(log_every))
            print(
                f"train[{tag}] update={update + 1}/{updates} "
                f"teacher_return={total_reward:.3f} "
                f"recent_motor_acc={float(np.mean(motor_accs[-window:])):.3f}",
                flush=True,
            )

    out_dir = Path(state_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("shared" if use_shared else "independent") + (
        "_uniform" if freeze_competition_query else ""
    )
    out_path = out_dir / f"competitive_infant_{tag}.pt"
    payload = {
        "format": FORMAT,
        "config": asdict(cfg),
        "vision_parent_dims": parent_dims,
        "model": _cpu_state_dict(model),
        "use_shared": bool(use_shared),
        "freeze_competition_query": bool(freeze_competition_query),
        "base_checkpoint_sha256": base_sha,
        "base_checkpoint_name": Path(base_checkpoint).name,
        "training_objective": "teacher_imitation_plus_world_prediction",
        "seed": int(seed),
        "updates": int(updates),
        "max_episode_steps": int(max_episode_steps),
        "corridor_lengths": [float(x) for x in corridor_lengths],
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "mean_teacher_return_last_100": float(np.mean(returns[-100:])) if returns else None,
        "mean_motor_accuracy_last_100": float(np.mean(motor_accs[-100:])) if motor_accs else None,
    }
    torch.save(payload, out_path)
    return {
        "output": str(out_path),
        "base_checkpoint_sha256": base_sha,
        "trainable_parameters": payload["trainable_parameters"],
        "mean_motor_accuracy_last_100": payload["mean_motor_accuracy_last_100"],
    }


def load_trained(path: str | Path, device: str | Any = "cpu") -> tuple[CompetitiveInfantBrain, dict]:
    device = _resolve_device(device)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {saved.get('format')!r}")
    cfg = InfantConfig(**saved["config"])
    model = CompetitiveInfantBrain(
        cfg,
        use_shared=bool(saved["use_shared"]),
        vision_parent_dims=saved.get("vision_parent_dims") or {},
    )
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, saved


def _aggregate(rows: list[dict]) -> dict:
    by_length = {}
    for L in CORRIDOR_LENGTHS:
        subset = [r for r in rows if math.isclose(r["corridor_length"], float(L), abs_tol=1e-8)]
        by_length[str(L)] = {
            "success_rate": float(np.mean([r["correct"] for r in subset])),
            "mean_steps": float(np.mean([r["steps"] for r in subset])),
            "world_context_mse": float(np.mean([r["world_context_mse"] for r in subset])),
            "selection_entropy": (
                float(np.mean([r["selection_entropy"] for r in subset if r["selection_entropy"] is not None]))
                if any(r["selection_entropy"] is not None for r in subset) else None
            ),
            "selection_disagreement": (
                float(np.mean([r["selection_disagreement"] for r in subset if r["selection_disagreement"] is not None]))
                if any(r["selection_disagreement"] is not None for r in subset) else None
            ),
        }
    return by_length


def _paired_summary(a: list[dict], b: list[dict], *, seed: int = 0) -> dict:
    key = lambda r: (int(r["seed"]), float(r["corridor_length"]))
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    common = sorted(set(aa) & set(bb))
    success = np.asarray([float(aa[k]["correct"]) - float(bb[k]["correct"]) for k in common])
    world = np.asarray([aa[k]["world_context_mse"] - bb[k]["world_context_mse"] for k in common])
    return {
        "n": len(common),
        "success_rate_difference": float(success.mean()) if len(success) else float("nan"),
        "success_rate_difference_95pct_bootstrap_ci": _bootstrap_ci(success, seed=seed),
        "world_context_mse_difference": float(world.mean()) if len(world) else float("nan"),
        "world_context_mse_difference_95pct_bootstrap_ci": _bootstrap_ci(world, seed=seed + 1),
    }


def evaluate(
    *,
    shared_checkpoint: str,
    independent_checkpoint: str | None = None,
    uniform_checkpoint: str | None = None,
    episodes_per_length: int = 100,
    seed_start: int = 91001,
    device: str | Any = "cpu",
    max_episode_steps: int = 120,
    output: str | None = None,
):
    device = _resolve_device(device)
    shared_model, shared_saved = load_trained(shared_checkpoint, device)
    cfg = shared_model.cfg
    expected_sha = shared_saved["base_checkpoint_sha256"]

    independent_model = independent_saved = None
    uniform_model = uniform_saved = None
    if independent_checkpoint:
        independent_model, independent_saved = load_trained(independent_checkpoint, device)
    if uniform_checkpoint:
        uniform_model, uniform_saved = load_trained(uniform_checkpoint, device)

    for name, saved in (("independent", independent_saved), ("uniform", uniform_saved)):
        if saved is not None and saved.get("base_checkpoint_sha256") != expected_sha:
            raise ValueError(f"{name} checkpoint was not grown from the same base checkpoint")
        if saved is not None and saved.get("config") != shared_saved.get("config"):
            raise ValueError(f"{name} checkpoint has a different InfantConfig")

    # This is the exact frozen base inside the trained shared wrapper, not a fresh
    # random InfantBrain.  It is also what disabled mode routes through.
    baseline_model = shared_model.base
    conditions: dict[str, tuple[Any, dict]] = {
        "baseline_frozen_infant_brain": (
            baseline_model, dict(use_competition=False, reset_hidden_every=None)
        ),
        "shared_selection": (
            shared_model, dict(use_competition=True, reset_hidden_every=None)
        ),
        "shared_hidden_reset_every_1": (
            shared_model, dict(use_competition=True, reset_hidden_every=1)
        ),
        "shared_competition_disabled": (
            shared_model, dict(use_competition=False, reset_hidden_every=None)
        ),
    }
    if independent_model is not None:
        conditions["independent_selection"] = (
            independent_model, dict(use_competition=True, reset_hidden_every=None)
        )
    if uniform_model is not None:
        conditions["shared_uniform_weights"] = (
            uniform_model, dict(use_competition=True, reset_hidden_every=None)
        )

    raw: dict[str, list[dict]] = {}
    aggregated = {}
    for name, (model, kwargs) in conditions.items():
        rows = []
        for L in CORRIDOR_LENGTHS:
            for i in range(int(episodes_per_length)):
                rows.append(
                    run_episode(
                        model,
                        int(seed_start) + i,
                        float(L),
                        device=device,
                        cfg=cfg,
                        max_episode_steps=max_episode_steps,
                        **kwargs,
                    )
                )
        raw[name] = rows
        aggregated[name] = _aggregate(rows)
        msg = ", ".join(
            f"L={L}:succ={aggregated[name][str(L)]['success_rate']:.2f}"
            for L in CORRIDOR_LENGTHS
        )
        print(f"{name}: {msg}", flush=True)

    # Disabled mode must be an exact behavioral control for the frozen base.
    base_rows = raw["baseline_frozen_infant_brain"]
    disabled_rows = raw["shared_competition_disabled"]
    parity = all(
        a["correct"] == b["correct"]
        and a["steps"] == b["steps"]
        and abs(a["world_context_mse"] - b["world_context_mse"]) <= 1e-12
        for a, b in zip(base_rows, disabled_rows)
    )
    if not parity:
        raise AssertionError("competition-disabled condition failed to reproduce the frozen base")

    paired = {
        "shared_minus_baseline": _paired_summary(raw["shared_selection"], base_rows, seed=1),
        "shared_minus_hidden_reset": _paired_summary(
            raw["shared_selection"], raw["shared_hidden_reset_every_1"], seed=2
        ),
    }
    if "independent_selection" in raw:
        paired["shared_minus_independent"] = _paired_summary(
            raw["shared_selection"], raw["independent_selection"], seed=3
        )
    if "shared_uniform_weights" in raw:
        paired["shared_minus_uniform"] = _paired_summary(
            raw["shared_selection"], raw["shared_uniform_weights"], seed=4
        )

    result = {
        "format": FORMAT,
        "base_checkpoint_sha256": expected_sha,
        "corridor_lengths": list(CORRIDOR_LENGTHS),
        "episodes_per_length": int(episodes_per_length),
        "max_episode_steps": int(max_episode_steps),
        "baseline_disabled_exact_parity": parity,
        "aggregated": aggregated,
        "paired_comparisons": paired,
        "interpretation_guardrails": [
            "Primary structural comparison is shared_selection versus independent_selection.",
            "Independent selection has more trainable selector parameters; a shared win is therefore conservative, while an independent win is capacity-confounded.",
            "A hidden-reset deficit supports load-bearing recurrent retention; no deficit means the task was solved without that recurrent carry-over.",
            "Uniform-weight performance separates learned selectivity from merely adding residual capacity.",
            "This experiment tests a functional competitive-broadcast mechanism. It does not establish subjective experience or consciousness.",
        ],
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train one experimental condition")
    p_train.add_argument("--base-checkpoint", required=True)
    p_train.add_argument("--condition", choices=("shared", "independent", "uniform"), required=True)
    p_train.add_argument("--updates", type=int, default=4000)
    p_train.add_argument("--seed", type=int, default=81501)
    p_train.add_argument("--learning-rate", type=float, default=1e-4)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--state-dir", default="state_competitive_infant")
    p_train.add_argument("--max-episode-steps", type=int, default=120)
    p_train.add_argument("--log-every", type=int, default=200)

    p_eval = sub.add_parser("evaluate", help="paired evaluation of trained conditions")
    p_eval.add_argument("--shared-checkpoint", required=True)
    p_eval.add_argument("--independent-checkpoint")
    p_eval.add_argument("--uniform-checkpoint")
    p_eval.add_argument("--episodes-per-length", type=int, default=100)
    p_eval.add_argument("--seed-start", type=int, default=91001)
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--max-episode-steps", type=int, default=120)
    p_eval.add_argument("--output", default="competitive_broadcast_results.json")

    args = parser.parse_args()
    if args.command == "train":
        use_shared = args.condition != "independent"
        uniform = args.condition == "uniform"
        print(json.dumps(train(
            base_checkpoint=args.base_checkpoint,
            use_shared=use_shared,
            updates=args.updates,
            seed=args.seed,
            learning_rate=args.learning_rate,
            freeze_competition_query=uniform,
            device=args.device,
            state_dir=args.state_dir,
            max_episode_steps=args.max_episode_steps,
            log_every=args.log_every,
        ), indent=2))
    else:
        result = evaluate(
            shared_checkpoint=args.shared_checkpoint,
            independent_checkpoint=args.independent_checkpoint,
            uniform_checkpoint=args.uniform_checkpoint,
            episodes_per_length=args.episodes_per_length,
            seed_start=args.seed_start,
            device=args.device,
            max_episode_steps=args.max_episode_steps,
            output=args.output,
        )
        print(json.dumps({
            "output": args.output,
            "baseline_disabled_exact_parity": result["baseline_disabled_exact_parity"],
            "paired_comparisons": result["paired_comparisons"],
        }, indent=2))


if __name__ == "__main__":
    _main()
