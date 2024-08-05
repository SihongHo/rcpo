from sandbox.cpo.envs.mujoco.gather.gather_noise_env import GatherNoiseEnv
from sandbox.cpo.envs.mujoco.gather.gather_adversary_env import GatherAdvEnv
from sandbox.cpo.envs.mujoco.ant_env import AntEnv
from rllab.core.serializable import Serializable
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

class AntGatherNoiseEnv(GatherNoiseEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6


class AntGatherAdvEnv(GatherAdvEnv):
    MODEL_CLASS = AntEnv
    ORI_IND = 6

    def __init__(self, adv_policy, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(AntGatherAdvEnv, self).__init__(adv_policy, *args, **kwargs)

    def __getstate__(self):
        d = super(AntGatherAdvEnv, self).__getstate__()
        # 保存 adv_policy 或其相关信息
        d['adv_policy_state'] = self.adv_policy.__getstate__()
        return d

    def __setstate__(self, d):
        # 重建 adv_policy
        adv_policy = GaussianMLPPolicy(,hidden_sizes=(64,32))
        adv_policy.__setstate__(d['adv_policy_state'])
        super(AntGatherAdvEnv, self).__setstate__(d)
        self.adv_policy = adv_policy

# 其中 MyPolicyClass 是创建 adv_policy 实例的类
