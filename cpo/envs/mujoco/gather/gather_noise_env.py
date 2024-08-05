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

class GatherNoiseEnv(GatherEnv):
    def __init__(self, noise_mean=0, noise_std=0.1, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.noise_std = noise_std  # 噪声的标准差
        self.noise_mean = noise_mean  # 噪声的均值
        super(GatherNoiseEnv, self).__init__(*args, **kwargs)

    def step(self, action):
        # 在这里添加噪声
        noise = np.random.normal(self.noise_mean, self.noise_std, size=action.shape)
        action_with_noise = action + noise

        # 使用带噪声的动作执行环境的步骤
        return super(GatherNoiseEnv, self).step(action_with_noise)