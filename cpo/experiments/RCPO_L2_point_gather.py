import sys

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Baseline
from sandbox.cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

# Environment
from sandbox.cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv
from sandbox.cpo.envs.mujoco.gather.adv_env import PointGatherAdvEnv, PointGatherNegEnv

# Policy optimization
# from sandbox.cpo.algos.safe.cpo import CPO
from sandbox.cpo.algos.safe.pcpo_L2 import PCPO_L2
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
# from sandbox.cpo.optimizers.conjugate_constraint_optimizer_pcpo_kl import ConjugateConstraintOptimizerPCPO_KL
from sandbox.cpo.safety_constraints.gather import GatherSafetyConstraint



ec2_mode = False


def run_task(*_):
        trpo_stepsize = 0.01
        trpo_subsample_factor = 0.2
        
        env = PointGatherEnv(apple_reward=10,bomb_cost=1,n_apples=2, activity_range=6)

        policy = GaussianMLPPolicy(env.spec,
                    hidden_sizes=(64,32)
                 )

        adv_policy = GaussianMLPPolicy(env.spec,
                    hidden_sizes=(64,32)
                 )
        
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    }
        )

        safety_baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    },
            target_key='safety_returns',
            )

        safety_constraint = GatherSafetyConstraint(max_value=0.1, baseline=safety_baseline)


        for i in range(100):
            if i % 2 == 0:
                noise_env = PointGatherAdvEnv(apple_reward=10,bomb_cost=1,n_apples=2, activity_range=6, adv_policy=adv_policy)
                algo_policy = PCPO_L2(
                    env=noise_env,
                    policy=policy,
                    baseline=baseline,
                    safety_constraint=safety_constraint,
                    safety_gae_lambda=1,
                    batch_size=50000,
                    max_path_length=15,
                    n_itr=2,
                    gae_lambda=0.95,
                    discount=0.995,
                    step_size=trpo_stepsize,
                    optimizer_args={'subsample_factor':trpo_subsample_factor},
                    #plot=True,
                )
                algo_policy.train()
            else:
                noise_env = PointGatherNegEnv(apple_reward=10,bomb_cost=1,n_apples=2, activity_range=6, adv_policy=policy)
                algo_adv_policy = PCPO_L2(
                    env=noise_env,
                    policy=adv_policy,
                    baseline=baseline,
                    safety_constraint=safety_constraint,
                    safety_gae_lambda=1,
                    batch_size=50000,
                    max_path_length=15,
                    n_itr=2,
                    gae_lambda=0.95,
                    discount=0.995,
                    step_size=trpo_stepsize,
                    optimizer_args={'subsample_factor':trpo_subsample_factor},
                    #plot=True,
                )
                algo_adv_policy.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    exp_prefix='RCPO_L2-PointGather',
    seed=1,
    mode = "ec2" if ec2_mode else "local"
    #plot=True
)
