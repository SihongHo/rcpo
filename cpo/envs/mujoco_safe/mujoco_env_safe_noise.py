from sandbox.cpo.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv

import numpy as np

class SafeMujocoNoiseEnv(SafeMujocoEnv, Serializable):
    # add gaussian noise to action
    def __init__(self, noise_mean=0, noise_std=1, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.noise_std = noise_std  
        self.noise_mean = noise_mean  
        super(SafeMujocoNoiseEnv, self).__init__(*args, **kwargs)
    
    def step(self, action):
        noise = np.random.normal(self.noise_mean, self.noise_std, size=action.shape)
        action_with_noise = action + noise

        return super(SafeMujocoNoiseEnv, self).step(action_with_noise)


class SafeHalfCheetahNoiseEnv(SafeMujocoNoiseEnv, Serializable):

    MODEL_CLASS = HalfCheetahEnv

class SafeSwimmerNoiseEnv(SafeMujocoNoiseEnv, Serializable):

    MODEL_CLASS = SwimmerEnv

class SafeWalker2DNoiseEnv(SafeMujocoNoiseEnv, Serializable):

    MODEL_CLASS = Walker2DEnv