import os
import sys
import statistics 
import numpy as np

from util import *
from stable_baseline.trpo_mpi import TRPO
from primitive_policy import PrimitivePolicy

def learn_primitive(env, config, save_path, env_name):
    print("Training Primitive : ", save_path.split("/")[-1])
    model = TRPO(PrimitivePolicy, env, max_kl=config.max_kl, cg_iters=config.cg_iters,
                 cg_damping=config.cg_damping, vf_stepsize=config.vf_stepsize, vf_iters=config.vf_iters, config=config, env_name=env_name, save_path=save_path)

    model.learn(total_timesteps=config.total_timesteps)
    return model.policy_pi

def load_primitive(env, config, path, env_name):
   policy = PrimitivePolicy(env=env, name="%s/pi" % env_name, ob_env_name=env_name, config=config, n_env=1)
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
        if(os.path.exists(path)):
            model = load_primitive(env, config, path, env_name)
        elif(env_name == config.prim_train):
            model = learn_primitive(env, config, path, env_name)
        
        if(model is not None):
            primitives.append(model)

    return primitives

def evaluate_primtive(env, policy, config):
    obs = env.reset()
    count = 0
    rewards = []
    while count < config.max_eval_iters:
        action, _states, _, _ = policy.step(obs)
        obs, reward, done, info = env.step(action)

        rewards.append(np.float(reward))
        if(config.render):
            env.render()
        if(done == True):
            obs = env.reset()
        count += 1
    print("Max Reward : ", max(rewards))
    print("Average Reward : ", statistics.mean(rewards))
