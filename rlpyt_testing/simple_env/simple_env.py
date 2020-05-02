import gym
from gym.spaces import Discrete
from gym.spaces import Box
import numpy as np

class SimpleEnv(gym.Env):
    '''
    This is a simple environment defining a a basic discrete optimization problem. It has ony 1 observation and 2 actions.
    Only one action yields a positive reward and that is unrelated to the observation.

    This environment should not be used with DQNAtari model (it requires rank 3 input (images). It should also not be used with
    MlpModel since that model expects input as floats and does not handle discrete variables (ReLu activation).
    This environment is not compatible with the CatDQNAgent because this only works with the CatDQN model which only accepts image input data.
    '''
    def __init__(self):
        self.action_space = Discrete(2)
        # DQNAgent model needs an observation space of rank 3 - 5 otherwise it will fail to determine lead dimension
        self.observation_space = Discrete(1)

    def close(self):
        return

    def reset(self):
        return np.array(0)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        obs = np.zeros(1, dtype=int)
        reward = -1
        done = False
        info = {}

        if action == 1:
            reward = 1

        return obs, reward, done, info


class SimpleEnvContinuous(gym.Env):
    '''
    This is a simple environment defining a continous optimization problem. The environment has only 1 observation
    and one action (ranging from 0-1). The action yields a positive reward in the range 0.5 - 0.75.
    '''
    def __init__(self):
        self.action_space = Box(0,1, shape=(1,), dtype=np.float64)
        self.observation_space = Box(-1,1, shape=(1,), dtype=np.float64)

    def close(self):
        return

    def reset(self):
        return np.array(0)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        obs = np.zeros(1, dtype=np.float64)
        reward = -1
        done = False
        info = {}

        if 0.5 < action < 0.75:
            reward = 100

        return obs, reward, done, info
