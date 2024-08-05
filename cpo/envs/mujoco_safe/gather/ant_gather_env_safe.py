from sandbox.cpo.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.cpo.envs.mujoco_safe.ant_env_safe import SafeAntEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = SafeAntEnv
    ORI_IND = 6
