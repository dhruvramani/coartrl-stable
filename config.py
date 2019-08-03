import os
import argparse

def str2bool(v):
    return v.lower() == 'true'

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]

def argparser():
    parser = argparse.ArgumentParser("Policy Stiching Framework",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='JacoToss-v1')
    parser.add_argument('--primitive_envs', type=str2list, default="JacoToss-v1,JacoHit-v1", help='Separated list \
                        of primitive envs eg. JacoToss-v1,JacoHit-v1')
    parser.add_argument('--primitive_paths', type=str2list, default="JacoToss.toss_coartl_prim, JacoHit.hit_coartl_prim", help='Separated list \
                        of model names inside primitive_dir loaded in order with primitive_envs \
                        eg. JacoToss.ICLR2019,JacoHit.ICLR2019')

    parser.add_argument('--is_coart', type=str2bool, default=False)
    parser.add_argument('--coart_alpha', type=float, default=10.0)
    parser.add_argument('--load_coart_path', type=str, default="./log/JacoToss.toss_coartl_prim")
    parser.add_argument('--is_train', type=str2bool, default=False)
    parser.add_argument('--total_timesteps', type=int, default=10000)
    parser.add_argument('--max_eval_iters', type=int, default=100)

    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--cg_damping', type=float, default=0.1)
    parser.add_argument('--vf_stepsize', type=float, default=1e-3)
    parser.add_argument('--vf_iters', type=int, default=5)

    parser.add_argument('--render', type=str2bool, default=True, help='Render frames')
    parser.add_argument('--policy_dir', type=str, default='./policies')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--debug', type=str2bool, default=False, help='See debugging info')

    args = parser.parse_args()
    return args