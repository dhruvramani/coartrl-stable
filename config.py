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
    parser.add_argument('--env', help='environment ID', type=str, default='JacoServe-v1')
    
    # --- Primitives ---
    parser.add_argument('--primitive_envs', type=str2list, default="JacoToss-v1,JacoHit-v1", help='Separated list \
                        of primitive envs eg. JacoToss-v1,JacoHit-v1')
    parser.add_argument('--primitive_paths', type=str2list, default="JacoToss.coartl_prim,JacoHit.coartl_prim", help='Separated list \
                        of model names inside primitive_dir loaded in order with primitive_envs \
                        eg. JacoToss.ICLR2019,JacoHit.ICLR2019')
    parser.add_argument('--prim_train', type=str, default='JacoToss-v1', 
        help="Specifies which primitive to train on the current run")
    parser.add_argument('--eval_primitives', type=str2bool, default=False)

    parser.add_argument('--primitive_num_hid_layers', type=int, default=2)
    parser.add_argument('--primitive_hid_size', type=int, default=32)
    parser.add_argument('--primitive_activation', type=str, default='tanh',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--primitive_fixed_var', type=str2bool, default=True)
    parser.add_argument('--primitive_include_acc', type=str2bool, default=False)
    parser.add_argument('--primitive_use_term', type=str2bool, default=True)

    # --- Coarticulation ---
    parser.add_argument('--is_coart', type=str2bool, default=True)
    parser.add_argument('--bridge_path', type=str, default="JacoServe.coartl_bridge")
    parser.add_argument('--sac_path', type=str, default="JacoServe.coartl_sac")
    parser.add_argument('--is_train', type=str2bool, default=False)
    parser.add_argument('--eval_all', type=str2bool, default=False)

    parser.add_argument('--bridge_kl', type=float, default=0.1)
    parser.add_argument('--stitch_naive', type=str2bool, default=False)

    # --- TRPO ---
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--cg_damping', type=float, default=0.1)
    parser.add_argument('--vf_stepsize', type=float, default=1e-3)
    parser.add_argument('--vf_iters', type=int, default=5)

    # --- SAC ---
    parser.add_argument('--sac_hid_size', type=int, default=32)
    parser.add_argument('--sac_num_hid_layers', type=int, default=2)
    parser.add_argument('--sac_activation', type=str, default='relu',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--sac_epochs', type=int, default=100)
    parser.add_argument('--sac_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--sac_batch_size', type=int, default=100)
    #parser.add_argument('--sac_max_ep_len', type=int, default=1000)
    parser.add_argument('--sac_replay_size', type=int, default=int(1e6))
    parser.add_argument('--sac_lr', type=float, default=1e-3)
    parser.add_argument('--sac_start_steps', type=int, default=10000, 
        help="Number of steps for uniform-random action selection, before running real policy. Helps exploration.")

    # --- Misc ---
    parser.add_argument('--num_rollouts', type=int, default=int(256))
    parser.add_argument('--total_timesteps', type=int, default=int(1e6))
    parser.add_argument('--max_eval_iters', type=int, default=int(1e3))
    parser.add_argument('--render', type=str2bool, default=False, help='Render frames')
    parser.add_argument('--policy_dir', type=str, default='./policies')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--debug', type=str2bool, default=False, help='See debugging info')

    args = parser.parse_args()
    return args
