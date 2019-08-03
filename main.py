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

from primitive import get_primitives
