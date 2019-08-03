import os
import sys

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.atari_wrappers import TransitionEnvWrapper

from config import argparser
from util import make_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

from primitive import get_primitives, evaluate_primtive


def run(config):
    env = make_env(config.env, config)
    primitives = get_primitives(config)

    print("Evaluating Primitives")
    for p_policy in primitives:
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