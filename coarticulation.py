import os
import sys
import statistics 
import numpy as np

from util import *
from stable_baseline.trpo_mpi import TRPO
from primitive_policy import PrimitivePolicy

from primitive import load_primitive
from main import evaluate_policy

def get_bridge_policy(env, primitives, config):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.bridge_path))
	
	if(os.path.exists(path)):
		model = load_primitive(env, config, path, config.bridge_path, config.env)

	if(config.is_train or model is None):
		print("Training Bridge Policy")
		trainer = TRPO(PrimitivePolicy, env, primitives=primitives, config=config, env_name=config.bridge_path, save_path=path, timesteps_per_batch=config.num_rollouts,
			max_kl=config.max_kl, cg_iters=config.cg_iters, cg_damping=config.cg_damping, vf_stepsize=config.vf_stepsize, vf_iters=config.vf_iters)
		trainer.learn(total_timesteps=config.total_timesteps)
		model = trainer.policy_pi

	if(config.eval_all):
		evaluate_policy(env, model, config)
	return model

def get_coartl_value(env, primitives, bridge_policy, config):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.value_path))
	if(os.path.exists(path)):
		pass
	
	if(config.is_train or model is None):
		pass
	return model

def get_coartl_policy(env, coartl_value, config):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.final_path))
	
	if(os.path.exists(path)):
		model = load_primitive(env, config, path, config.final_path)

	if(config.is_train or model is None):
		pass

	return model
