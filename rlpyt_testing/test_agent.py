from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.agents.dqn.catmlpdqn_agent import CatMlpDqnAgent
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.logging import logger
from simple_env.simple_env import SimpleEnv
import torch
from collections import namedtuple

def f(*args, **kwargs):
    return GymEnvWrapper(SimpleEnv())

def test_agent(weightsfile, run_nr, epochs, logname):
    assert(weightsfile)

    params = torch.load(weightsfile)

    sample = SerialSampler(
        EnvCls=f,
        env_kwargs={},
        batch_T=100,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=10e3,
        eval_max_trajectories=5,
        eval_env_kwargs={}
    )

    agent = CatMlpDqnAgent(
        eps_init=0,
        eps_final=0
    )
    #load weights
    agent.initialize(f().spaces)
    agent.model.load_state_dict(params['agent_state_dict']['model'])

    config = {}
    name = logname + SimpleEnv.__class__.__name__
    log_dir = logname
    with logger_context(log_dir, run_nr, name, config, snapshot_mode='last', use_summary_writer=True):
        logger.log('Starting run...')
        sample.initialize(agent)
        running_mean_reward=running_max_reward=running_min_reward = 0

        for i in range(epochs):
            samples, infos = sample.obtain_samples(i)
            mean_reward = samples.env.reward.mean()
            max_reward = samples.env.reward.max()
            min_reward = samples.env.reward.min()
            logger.log(f'Test {i}\t: reward mean={mean_reward:.2f}, \t max={max_reward}, \t min={min_reward}')
            running_mean_reward += mean_reward
            running_max_reward = max([running_max_reward, max_reward])
            running_min_reward = min([running_min_reward, min_reward])
            logger.log(f'running\t: reward mean={running_mean_reward/i:.2f}, \t max={running_max_reward}, \t min={running_min_reward}')

        logger.log('Got samples from agent in environment...')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', help='weights file to use', type=str)
    parser.add_argument('--run', help='run number (for logging)', type=int, default=0)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=30)
    parser.add_argument('--logname', help='Name to identify logfile', type=str, default='test_agent')
    args = parser.parse_args()

    test_agent(
        weightsfile=args.weights,
        run_nr=args.run,
        epochs=args.epochs,
        logname=args.logname
    )
