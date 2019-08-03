import os
import sys

from util import make_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

def learn_primitive(env, config, save_path):
    model = TRPO(MlpPolicy, env, max_kl=config.max_kl, cg_iters=config.cg_iters,
                 cg_damping=config.cg_damping, vf_stepsize=config.vf_stepsize, vf_iters=config.vf_iters)

    model.learn(total_timesteps=config.total_timesteps)
    model.save(save_path)
    return model

def get_primitives(config):
    primitives = []
    for i, p_path in enumerate(config.primitive_paths):
        path = os.path.expanduser(os.path.join(config.policy_dir, p_path))
        if(os.path.exists(path)):
            model = TRPO.load(path)
        else :
            env = make_env(config.primitive_envs[i], config)
            model = learn_primitive(env, config, path)
        primitives.append(model)

    return primitives