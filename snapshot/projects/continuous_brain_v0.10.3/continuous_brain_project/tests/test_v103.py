from types import SimpleNamespace

import torch

from continuous_brain import __version__
from continuous_brain.planning_v103 import (
    _hierarchical_action,
    _hierarchical_option_choice,
    _rollout_option,
)


class FakeOptions:
    def __init__(self, action_for_option=None, selector=None, terminate=-100.0):
        self.action_for_option = action_for_option or {0: 1, 1: 2, 2: 4}
        self.selector = selector if selector is not None else torch.tensor([[4.0, 3.0, 2.0]])
        self.terminate = float(terminate)

    def selector_logits(self, latent):
        return self.selector.to(latent.device).expand(latent.shape[0], -1)

    def action_logits(self, latent, option):
        out = torch.full((latent.shape[0], 5), -10.0, device=latent.device)
        for i, o in enumerate(option.tolist()):
            out[i, self.action_for_option[int(o)]] = 10.0
        return out

    def termination_logit(self, latent, option):
        return torch.full((latent.shape[0],), self.terminate, device=latent.device)

    def action_value(self, latent, option, action_onehot):
        # Stable deterministic tail values; the exact magnitude is not important.
        return 0.1 + 0.01 * option.float()


class FakePlanner:
    def __init__(self):
        self.actions = []

    def encode_state(self, spatial, context):
        return spatial

    def step(self, latent, action_onehot):
        actions = action_onehot.argmax(dim=-1)
        self.actions.extend(actions.tolist())
        # Reward option-policy motion equally so selector order breaks exact ties.
        return {
            "latent": latent + 0.01,
            "reward": torch.full((latent.shape[0],), 0.05, device=latent.device),
        }


class FakeModel:
    def __init__(self, options=None, planner=None):
        self.cfg = SimpleNamespace(action_dim=5, option_count=3)
        self.options = options or FakeOptions()
        self.planning = planner or FakePlanner()


def test_version_is_v103():
    assert __version__ == "0.10.3"


def test_macro_rollout_uses_option_policy_for_every_primitive_step():
    planner = FakePlanner()
    model = FakeModel(planner=planner)
    z = torch.zeros(1, 2)
    result = _rollout_option(
        model, z, 0, max_option_steps=4, discount=0.95,
        min_commit=3, terminate_threshold=0.5,
    )
    assert result["actions"] == [1, 1, 1, 1]
    assert planner.actions == [1, 1, 1, 1]


def test_active_option_owns_primitive_control_between_boundaries():
    planner = FakePlanner()
    model = FakeModel(planner=planner)
    result = _hierarchical_action(
        model, torch.zeros(1, 2), torch.zeros(1, 7),
        current_option=2, option_age=1,
        option_depth=2, selector_topk=2, beam_width=4,
        max_option_steps=4, discount=0.95,
        min_commit=3, terminate_threshold=0.5,
    )
    assert result["high_level_decision"] is False
    assert result["option"] == 2
    assert result["action"] == 4
    assert planner.actions == []  # no recurrent search while intention is active


def test_high_level_search_expands_options_not_unrestricted_actions():
    planner = FakePlanner()
    options = FakeOptions(action_for_option={0: 1, 1: 2, 2: 4})
    model = FakeModel(options=options, planner=planner)
    result = _hierarchical_option_choice(
        model, torch.zeros(1, 2), option_depth=1,
        selector_topk=2, beam_width=4, max_option_steps=3,
        discount=0.95, min_commit=3, terminate_threshold=0.5,
    )
    # Selector top-2 are options 0 and 1, whose policies emit actions 1 and 2.
    assert set(planner.actions).issubset({1, 2})
    assert 1 in planner.actions and 2 in planner.actions
    assert result["chosen_option"] in {0, 1}
    assert result["imagined_options"]
