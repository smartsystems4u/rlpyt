
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.catmlp import CatMlpModel
from rlpyt.agents.dqn.atari.mixin import AtariMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["p"])


class CatMlpDqnAgent(DqnAgent):
    """Agent for Categorical DQN algorithm for use with an MLP model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.n_atoms = self.model_kwargs["n_atoms"]

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        self.n_atoms = env_spaces.action.n
        self.model_kwargs = {'input_size': env_spaces.observation.n, 'hidden_sizes': self.n_atoms, 'output_size':env_spaces.action.n}
        self.ModelCls = CatMlpModel
        super().initialize(env_spaces, share_memory )
        # Overwrite distribution.
        self.n_atoms = self.model._n_atoms
        # self.distribution = EpsilonGreedy(dim=env_spaces.action.n,
        #     z=torch.linspace(-1, 1, self.n_atoms))  # z placeholder for init.
        self.distribution = EpsilonGreedy(dim=env_spaces.action.n)

    def give_V_min_max(self, V_min, V_max):
        self.V_min = V_min
        self.V_max = V_max
        self.distribution.set_z(torch.linspace(V_min, V_max, self.n_atoms))

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        p = self.model(*model_inputs)
        # p = self.model(observation.float())
        p = p.cpu()
        action = self.distribution.sample(p)
        agent_info = AgentInfo(p=p)  # Only change from DQN: q -> p.
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)
