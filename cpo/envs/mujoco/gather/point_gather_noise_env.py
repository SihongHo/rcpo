from sandbox.cpo.envs.mujoco.gather.gather_noise_env import GatherNoiseEnv
from sandbox.cpo.envs.mujoco.gather.gather_adversary_env import GatherAdvEnv
from sandbox.cpo.envs.mujoco.point_env import PointEnv


class PointGatherNoiseEnv(GatherNoiseEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2

class PointGatherAdvEnv(GatherAdvEnv): 
    
    MODEL_CLASS = PointEnv
    ORI_IND = 2