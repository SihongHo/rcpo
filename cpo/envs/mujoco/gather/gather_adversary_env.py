import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from ctypes import byref

import numpy as np
import theano

from rllab import spaces
from rllab.misc import logger
from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.envs.mujoco.gather.embedded_viewer import EmbeddedViewer
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.mujoco_py import MjViewer, MjModel, mjcore, mjlib, \
    mjextra, glfw


from sandbox.cpo.envs.mujoco.point_env import PointEnv
from sandbox.cpo.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.cpo.envs.mujoco.gather.gather_env import GatherViewer

APPLE = 0
BOMB = 1

class GatherAdvEnv(GatherEnv):
    def __init__(self, adv_policy, noise_scale=0.001, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.adv_policy = adv_policy  # 使用 GaussianMLPPolicy
        self.noise_scale = noise_scale  # 噪声的限制
        super(GatherAdvEnv, self).__init__(*args, **kwargs)

    def step(self, action):
        # 使用当前状态获取噪声
        current_state = self.get_current_obs()
        noise, _ = self.adv_policy.get_action(current_state)
        noise = np.clip(noise, -action*self.noise_scale, action*self.noise_scale)
        # 将噪声添加到动作
        action_with_noise = action + noise

        # 使用带噪声的动作执行环境的步骤
        return super(GatherAdvEnv, self).step(action_with_noise)
