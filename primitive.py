import os
import sys
import statistics 
import numpy as np

from util import *
from stable_baseline.trpo_mpi import TRPO
from primitive_policy import PrimitivePolicy

def learn_primitive(env, config, save_path, env_name):
    print("Training Primitive : ", save_path.split("/")[-1])
    model = TRPO(PrimitivePolicy, env, max_kl=config.max_kl, cg_iters=config.cg_iters, timesteps_per_batch=config.num_rollouts,
                 cg_damping=config.cg_damping, vf_stepsize=config.vf_stepsize, vf_iters=config.vf_iters, config=config, env_name=env_name, save_path=save_path)

    model.learn(total_timesteps=config.total_timesteps)
    return model.policy_pi

def load_primitive(env, config, path, env_name, ob_env_name=None):
   if(ob_env_name is None):
      ob_env_name = env_name
   policy = PrimitivePolicy(env=env, name="%s/pi" % env_name, ob_env_name=ob_env_name, config=config, n_env=1)
   policy_vars = policy.get_variables()
   policy_path = load_model(path, policy_vars)
   return policy

def get_primitives(config):
    primitives = []
    for i, p_path in enumerate(config.primitive_paths):
        path = os.path.expanduser(os.path.join(config.policy_dir, p_path))
        env_name = config.primitive_envs[i]
        env = make_env(config.primitive_envs[i], config)
        model = None

        if(os.path.exists(path) and not config.train_primitives):
            model = load_primitive(env, config, path, env_name)
        # elif(env_name == config.prim_train):
        elif(config.train_primitives):
            model = learn_primitive(env, config, path, env_name)
        
        if(model is not None):
            primitives.append(model)

    return primitives
