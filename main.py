import os
import sys

import tensorflow as tf
import baselines.common.tf_util as tf_util
from baselines.common import set_global_seeds

from config import argparser
from util import make_env

from primitive import get_primitives
from coarticulation import *

from sac.core import PrimitivePolicySAC

def run(config):
    sess = tf_util.single_threaded_session(gpu=False)
    sess.__enter__()

    env = make_env(config.env, config)
    primitives = get_primitives(config)

    if(config.eval_primitives):
        for i, p_policy in enumerate(primitives):
            print("Evaluating Primitive for Env. ", config.primitive_envs[i])
            prim_env = make_env(config.primitive_envs[i], config)
            evaluate_policy(prim_env, p_policy, config)
            prim_env.close()

    if(config.is_coart):
        bridge_policy = None
        if(config.train_bridge):
            bridge_policy = get_bridge_policy(env, primitives, config)
        coartl_sac = get_coartl_sac(env, config, primitives, bridge_policy)
        print("Evaluating SAC for Env. ", config.env)
        evaluate_policy(env, coartl_sac, config)

    env.close()

def evaluate_policy(env, policy, config):
    obs = env.reset()
    count = 0
    rewards = []
    while count < config.max_eval_iters:
        if(isinstance(policy, PrimitivePolicySAC)):
            action, _ = policy.step(obs)
        else:
            action, _, _, _ = policy.step(obs)
        obs, reward, done, info = env.step(action)

        rewards.append(np.float(reward))
        env.render()
        if(done == True):
            obs = env.reset()
        count += 1
    print("Max Reward : ", max(rewards))
    print("Average Reward : ", statistics.mean(rewards))

def main():
    args = argparser()
    run(args)

if __name__ == '__main__':
    main()
