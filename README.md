# RCPO: Robust Constrained Policy Optimization

Code of ICML paper: Constrained Reinforcement Learning Under Model Mismatch, https://arxiv.org/pdf/2405.01327

## DRL
The code of deep RL version (in rcpo/cpo) is modified from CPO and PCPO. You may check https://github.com/jachiam/cpo, and follow the instruction to set up the environment for gather and circle tasks. Note that you may need to request the license for MuJoCo.

Once set up the environment, you can run experiments using 

``` python rllab/sandbox/cpo/experiments/RCPO_KL_point_gather.py ```

``` python rllab/sandbox/cpo/experiments/RCPO_L2_point_gather.py ```
