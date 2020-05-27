from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from deep_sea_treasure_env.deep_sea_treasure_env import DeepSeaTreasureEnv
from collections import namedtuple
from rlpyt.algos.dqn.r2d1 import R2D1
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
from rlpyt.models.dqn.catmlp_r2d1_model import CatMlpR2D1Model

def f(*args, **kwargs):
    return GymEnvWrapper(DeepSeaTreasureEnv())

def build_and_train(run_nr = 0):
    sample = SerialSampler(
        EnvCls=f,
        env_kwargs={},
        batch_T=50,
        batch_B=10,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=10e3,
        # eval_max_trajectories=5,
        eval_env_kwargs={}
    )

    algo = R2D1(
        eps_steps=35e6
    ) #default values
    spaces = namedtuple('spaces', ['actionspace', 'observationspace'])
    agent = R2d1Agent(
                                ModelCls=CatMlpR2D1Model,
                                model_kwargs = {'input_size':3,
                                                'hidden_sizes': 20,
                                                "output_size": 4})

    runner = MinibatchRlEval(
        algo = algo,
        agent= agent,
        sampler = sample,
        n_steps = 40e6,
        log_interval_steps = 10e3
    )
    config = {}
    name = "deep_sea_treasure_r2d1: " + DeepSeaTreasureEnv.__class__.__name__
    log_dir = "deep_sea_treasure_env"
    with logger_context(log_dir, run_nr, name, config, snapshot_mode='last', use_summary_writer=True):
        runner.train()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run', help='run number (for logging)', type=int, default=0)
    args = parser.parse_args()

    build_and_train(
        run_nr= args.run
    )
