from sandbox.cpo.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.base import Step

import numpy as np

class SafeMujocoAdvEnv(SafeMujocoEnv, Serializable):
    # add adv_policy to action
    def __init__(self, adv_policy, noise_scale=0.001, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.adv_policy = adv_policy
        self.noise_scale = noise_scale  # 噪声的限制
        super(SafeMujocoAdvEnv, self).__init__(*args, **kwargs)
    
    def step(self, action):
        adv_action = self.adv_policy.get_action(self.get_current_obs())[0]
        adv_action = np.clip(adv_action, -action*self.noise_scale, action*self.noise_scale)

        return super(SafeMujocoAdvEnv, self).step(action + adv_action)

class SafeMujocoNegEng(SafeMujocoAdvEnv, Serializable):
    # reture negative reward
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(SafeMujocoNegEng, self).__init__(*args, **kwargs)
    
    def step(self, action):
        adv_action = self.adv_policy.get_action(self.get_current_obs())[0]
        action = np.clip(action, -adv_action*self.noise_scale, adv_action*self.noise_scale)
        _, reward, done, info = self.wrapped_env.step(action + adv_action)
        next_obs = self.get_current_obs()

        if self._circle_mode:
            pos = self.wrapped_env.get_body_com("torso")
            vel = self.wrapped_env.get_body_comvel("torso")
            dt = self.wrapped_env.model.opt.timestep
            x, y = pos[0], pos[1]
            dx, dy = vel[0], vel[1]
            reward = -y * dx + x * dy
            reward /= (1 + np.abs( np.sqrt(x **2 + y **2) - self._target_dist))

        if self._nonlinear_reward:
            reward *= np.abs(reward)
        self._step += 1
        if self._max_path_length_range is not None:
            if self._step > self._last_step:
                done = True
        return Step(next_obs, -reward, done, **info)

class SafeHalfCheetahAdvEnv(SafeMujocoAdvEnv, Serializable):

    MODEL_CLASS = HalfCheetahEnv

class SafeSwimmerAdvEnv(SafeMujocoAdvEnv, Serializable):

    MODEL_CLASS = SwimmerEnv

class SafeWalker2DAdvEnv(SafeMujocoAdvEnv, Serializable):

    MODEL_CLASS = Walker2DEnv

class SafeHalfCheetahNegEng(SafeMujocoNegEng, Serializable):

    MODEL_CLASS = HalfCheetahEnv

class SafeSwimmerNegEng(SafeMujocoNegEng, Serializable):

    MODEL_CLASS = SwimmerEnv

class SafeWalker2DNegEng(SafeMujocoNegEng, Serializable):

    MODEL_CLASS = Walker2DEnv