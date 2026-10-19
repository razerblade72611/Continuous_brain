import copy
import torch

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import InfantBrain, InfantConfig
from continuous_brain.parented_train import _configure_world_only


def _direct_action(obs):
    dx, dy = float(obs[2]), float(obs[3])
    if abs(dx) > abs(dy):
        return 4 if dx > 0 else 3
    return 1 if dy > 0 else 2


def test_detour_world_makes_direct_greedy_collide():
    env = DetourWorld(seed=9001, max_episode_steps=80)
    obs = env.reset()
    collided = False
    for _ in range(40):
        obs, _, done, info = env.step(_direct_action(obs))
        collided = collided or bool(info.get("collision"))
        if done:
            break
    assert collided


def test_world_only_freezes_all_non_world_parameters():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    names = _configure_world_only(model)
    assert names
    assert all(name.startswith("world.") for name in names)
    assert all(p.requires_grad for p in model.world.parameters())
    assert all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("world."))


def test_world_only_optimizer_step_does_not_change_motor():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    _configure_world_only(model)
    before = {k: v.detach().clone() for k, v in model.motor.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    workspace = torch.randn(2, model.cfg.bus_dim)
    action = torch.nn.functional.one_hot(torch.tensor([1, 4]), model.cfg.action_dim).float()
    base_bus = torch.randn(2, model.cfg.bus_dim)
    base_context = torch.randn(2, model.cfg.dynamics_context_dim)
    out = model.world(workspace, action, base_bus=base_bus, base_context=base_context)
    loss = out["reward"].square().mean() + out["next_context"].square().mean()
    loss.backward(); opt.step()
    after = model.motor.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k])
