from rllab.core.serializable import Serializable
import numpy as np
from rllab.envs.base import Env, Step
from sandbox.cpo.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from sandbox.cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv

APPLE = 0

class AntGatherAdvEnv(AntGatherEnv, Serializable):
    def __init__(self, adv_policy, noise_scale=0.001, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.adv_policy = adv_policy  # 使用 GaussianMLPPolicy
        self.noise_scale = noise_scale  # 噪声的限制
        super(AntGatherAdvEnv, self).__init__(*args, **kwargs)


    def step(self, action):
        # 使用当前状态获取噪声
        current_state = self.get_current_obs()
        noise, _ = self.adv_policy.get_action(current_state)
        noise = np.clip(noise, -action*self.noise_scale, action*self.noise_scale)
        # 将噪声添加到动作
        action_with_noise = action + noise

        # 使用带噪声的动作执行环境的步骤
        return super(AntGatherAdvEnv, self).step(action_with_noise)

class AntGatherNegEnv(AntGatherEnv, Serializable):
    def __init__(self, adv_policy, noise_scale=0.001, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.adv_policy = adv_policy  # 使用 GaussianMLPPolicy
        self.noise_scale = noise_scale  # 噪声的限制
        super(AntGatherNegEnv, self).__init__(*args, **kwargs)
    
    def step(self, action):
        # 使用当前状态获取噪声
        current_state = self.get_current_obs()
        noise, _ = self.adv_policy.get_action(current_state)
        action = np.clip(action, -noise*self.noise_scale, noise*self.noise_scale)

        # 将噪声添加到动作
        action_with_noise = action + noise

        # 使用带噪声的动作执行环境的步骤
        _, _, done, info = self.inner_env.step(action_with_noise)
        info['apples'] = 0
        info['bombs'] = 0
        if done:
            return Step(self.get_current_obs(), 10, done, **info)  # 将奖励变为正值
        com = self.inner_env.get_body_com("torso")
        x, y = com[:2]
        reward = 0
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward += self.apple_reward
                    info['apples'] = 1
                else:
                    reward -= self.bomb_cost
                    info['bombs'] = 1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        reward = -reward  # 将奖励转换为负值
        return Step(self.get_current_obs(), reward, done, **info)

class PointGatherAdvEnv(PointGatherEnv, Serializable):
    def __init__(self, adv_policy, noise_scale=0.001, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.adv_policy = adv_policy  # 使用 GaussianMLPPolicy
        self.noise_scale = noise_scale  # 噪声的限制
        super(PointGatherAdvEnv, self).__init__(*args, **kwargs)


    def step(self, action):
        # 使用当前状态获取噪声
        current_state = self.get_current_obs()
        noise, _ = self.adv_policy.get_action(current_state)
        noise = np.clip(noise, -action*self.noise_scale, action*self.noise_scale)
        # 将噪声添加到动作
        action_with_noise = action + noise

        # 使用带噪声的动作执行环境的步骤
        return super(PointGatherAdvEnv, self).step(action_with_noise)

class PointGatherNegEnv(PointGatherEnv, Serializable):
    def __init__(self, adv_policy, noise_scale=0.001, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.adv_policy = adv_policy  # 使用 GaussianMLPPolicy
        self.noise_scale = noise_scale  # 噪声的限制
        super(PointGatherNegEnv, self).__init__(*args, **kwargs)
    
    def step(self, action):
        # 使用当前状态获取噪声
        current_state = self.get_current_obs()
        noise, _ = self.adv_policy.get_action(current_state)
        action = np.clip(action, -noise*self.noise_scale, noise*self.noise_scale)

        # 将噪声添加到动作
        action_with_noise = action + noise

        # 使用带噪声的动作执行环境的步骤
        _, _, done, info = self.inner_env.step(action_with_noise)
        info['apples'] = 0
        info['bombs'] = 0
        if done:
            return Step(self.get_current_obs(), 10, done, **info)  # 将奖励变为正值
        com = self.inner_env.get_body_com("torso")
        x, y = com[:2]
        reward = 0
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward += self.apple_reward
                    info['apples'] = 1
                else:
                    reward -= self.bomb_cost
                    info['bombs'] = 1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        reward = -reward  # 将奖励转换为负值
        return Step(self.get_current_obs(), reward, done, **info)
