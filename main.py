import os
import sys

import baselines.common.tf_util as tf_util
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.atari_wrappers import TransitionEnvWrapper

from config import argparser
from util import make_env

from primitive import get_primitives, evaluate_primtive

def run(config):
    sess = tf_util.single_threaded_session(gpu=False)
    sess.__enter__()
    env = make_env(config.env, config)
    primitives = get_primitives(config)

    if(config.eval_primitives):
        for i, p_policy in enumerate(primitives):
            print("Evaluating Primitive for Env. ", config.primitive_envs[i])
            evaluate_primtive(env, p_policy, config)

    if(config.is_coart):
        # NOTE : Implement Algo here
        pass

    env.close()


def main():
    args = argparser()
    run(args)

if __name__ == '__main__':
    main()
