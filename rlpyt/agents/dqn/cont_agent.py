from rlpyt.agents.base import AgentStep, BaseAgent
from rlpyt.agents.dqn.dqn_agent import DqnAgent, AgentInfo
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.models.catmlp import CatMlpModel
import torch
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.buffer import buffer_to


class ContinuousDqnAgent(EpsilonGreedyAgentMixin,BaseAgent):
    '''
    Inherited from BaseAgent instead of DqnAgent because of reliance on gym.space.n.
    ToDo: Properly add epsilongreedy action selection
    '''

    def __init__(self, ModelCls=CatMlpModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def __call__(self, observation, prev_action, prev_reward):
        """Returns Q-values for states/observations (with grad)."""
        # prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q = self.model(*model_inputs)
        return q.cpu()

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        self.n_atoms = 2 * len(env_spaces.action.sample().flatten())
        self.model_kwargs = {'input_size': len(env_spaces.observation.sample().flatten()), 'hidden_sizes': self.n_atoms, 'output_size':len(env_spaces.action.sample().flatten())}
        self.ModelCls = CatMlpModel
        super().initialize(env_spaces, share_memory )
        self.target_model = self.ModelCls(**self.env_model_kwargs,
                                          **self.model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        # EpsilonGreedy assumes we need to select an action based on argmax over actionspace
        # In our case we don't have this mapping so we will do this ourselves in step()
        # TODO: Check where the agent.distribution is used elsewhere
        self.distribution = EpsilonGreedy(dim=len(env_spaces.action.sample().flatten()))
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.target_model.to(self.device)

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and uses model output directly (no grad)"""
        # prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q = self.model(*model_inputs)
        # q = self.model(observation.float())
        q = q.cpu()
        # action = self.distribution.sample(q)

        if torch.rand(1) < self.distribution._epsilon:
            action = torch.rand(q.shape)
        else:
            action = q

        agent_info = AgentInfo(q=q)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, observation, prev_action, prev_reward):
        """Returns the target Q-values for states/observations."""
        # prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_q = self.target_model(*model_inputs)
        return target_q.cpu()

    def update_target(self, tau=1):
        """Copies the model parameters into the target model."""
        update_state_dict(self.target_model, self.model.state_dict(), tau)

