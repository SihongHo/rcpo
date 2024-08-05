from sandbox.cpo.envs.mujoco.gather.gather_noise_env import GatherNoiseEnv
from sandbox.cpo.envs.mujoco.ant_env import AntEnv
from rllab.core.serializable import Serializable

class AntGatherNoiseEnv(GatherNoiseEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
