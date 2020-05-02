from rlpyt.samplers.serial.sampler import SerialSampler
#from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.agents.dqn.cont_agent import ContinuousDqnAgent
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.cont_dqn import ContDQN
#from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel
from rlpyt.models.dqn.atari_catdqn_model import DistributionalHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.catmlp import CatMlpModel
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from simple_env.simple_env import SimpleEnvContinuous
from collections import namedtuple

def f(*args, **kwargs):
    return GymEnvWrapper(SimpleEnvContinuous())

def build_and_train():
    sample = SerialSampler(
        EnvCls=f,
        env_kwargs={},
        batch_T=4,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=10e3,
        eval_max_trajectories=5,
        eval_env_kwargs={}
    )
    # algo = DQN(min_steps_learn=10e3) #run with defaults
    algo = ContDQN(min_steps_learn=10e3)
    # spaces = namedtuple('spaces', ['actionspace', 'observationspace'])
    agent = ContinuousDqnAgent()
    runner = MinibatchRlEval(
        algo = algo,
        agent= agent,
        sampler = sample,
        n_steps = 40e5,
        log_interval_steps = 10e3
    )
    config = {}
    name = "Simple_test_continuous: " + SimpleEnvContinuous.__class__.__name__
    log_dir = "simple_test_continuous"
    with logger_context(log_dir, 12, name, config, snapshot_mode='last', use_summary_writer=True):
        runner.train()

if __name__ == "__main__":
    build_and_train()
