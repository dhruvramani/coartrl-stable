import os
import sys
import statistics 
import numpy as np

from util import make_env

#from stable_baseline.policies import MlpPolicy
from stable_baseline.trpo_mpi import TRPO
from primitive_policy import PrimitivePolicy

def learn_primitive(env, config, save_path, env_name):
    print("Training Primitive : ", save_path.split("/")[-1])
    model = TRPO(PrimitivePolicy, env, max_kl=config.max_kl, cg_iters=config.cg_iters,
                 cg_damping=config.cg_damping, vf_stepsize=config.vf_stepsize, vf_iters=config.vf_iters, config=config, env_name=env_name)

    model.learn(total_timesteps=config.total_timesteps)
    model.save(save_path)
    return model

def get_primitives(config):
    primitives = []
    for i, p_path in enumerate(config.primitive_paths):
        path = os.path.expanduser(os.path.join(config.policy_dir, p_path))
        env_name = config.primitive_envs[i]
        if(os.path.exists(path)):
            model = TRPO.load(path)
        elif(env_name == config.prim_train):
            env = make_env(config.primitive_envs[i], config)
            model = learn_primitive(env, config, path, env_name)
        primitives.append(model)

    return primitives

def evaluate_primtive(env, policy, config):
    obs = env.reset()
    count = 0
    rewards = []
    while count < config.max_eval_iters:
        action, _states = policy.predict(obs)
        obs, reward, done, info = env.step(action)

        rewards.append(np.float(reward))
        if(config.render):
            env.render()
        if(done == True):
            obs = env.reset()
        count += 1
    print("Max Reward : ", max(rewards))
    print("Average Reward : ", statistics.mean(rewards))
