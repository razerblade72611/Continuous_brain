import torch

from continuous_brain.agent import ContinuousAgent
from continuous_brain.config import BrainConfig
from continuous_brain.environment import ForagingWorld


class MaskedForaging(ForagingWorld):
    def valid_action_indices(self):
        return (0, 2, 3)


def test_agent_greedy_policy_respects_environment_mask(tmp_path):
    env = MaskedForaging()
    agent = ContinuousAgent(env, BrainConfig(epsilon_start=0.0, epsilon_end=0.0), tmp_path)
    try:
        def q_values(_hidden):
            # Invalid action 1 is deliberately highest.
            return torch.tensor([[0.0, 100.0, 3.0, 2.0, 50.0]], device=agent.device)
        agent.net.q_values = q_values
        action = agent._neural_action(agent.net.initial_state(1, agent.device))
        assert action == 2
    finally:
        agent.episodic.close()
        env.close()


def test_agent_random_policy_samples_only_valid_actions(tmp_path):
    env = MaskedForaging()
    agent = ContinuousAgent(env, BrainConfig(epsilon_start=1.0, epsilon_end=1.0), tmp_path)
    try:
        hidden = agent.net.initial_state(1, agent.device)
        observed = {agent._neural_action(hidden) for _ in range(200)}
        assert observed <= {0, 2, 3}
        assert 1 not in observed and 4 not in observed
    finally:
        agent.episodic.close()
        env.close()
