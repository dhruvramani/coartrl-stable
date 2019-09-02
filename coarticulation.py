import os
import sys
import statistics 
import numpy as np

from sac.sac import SAC
from sac.core import PrimitivePolicySAC

from stable_baseline.trpo_mpi import TRPO

from util import *
from primitive_policy import PrimitivePolicy

from primitive import load_primitive
from main import evaluate_policy

def get_bridge_policy(env, primitives, config):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.bridge_path))
	
	if(os.path.exists(path)):
		printstar("Loading Bridge Policy")
		model = load_primitive(env, config, path, config.bridge_path, config.env)

	if(config.is_train or model is None):
		printstar("Training Bridge Policy")
		trainer = TRPO(PrimitivePolicy, env, primitives=primitives, config=config, env_name=config.bridge_path, save_path=path, timesteps_per_batch=config.num_rollouts,
			max_kl=config.max_kl, cg_iters=config.cg_iters, cg_damping=config.cg_damping, vf_stepsize=config.vf_stepsize, vf_iters=config.vf_iters)
		trainer.learn(total_timesteps=config.total_timesteps)
		model = trainer.policy_pi

	if(config.eval_all):
		evaluate_policy(env, model, config)
	return model

def get_coartl_sac(env, config, primitives=None, bridge_policy=None):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.sac_path))
	if(os.path.exists(path)):
		printstar("Loading SAC")
		model = load_sac(env, config, path) 
	
	if(config.is_train or model is None):
		printstar("Training SAC")
		test_env = make_env(config.env)
		model = SAC(env, test_env, path, config, primitives=primitives, bridge_policy=bridge_policy)

	if(config.eval_all):
		evaluate_policy(env, model, config)

	return model

def load_sac(env, config, path):
	policy = PrimitivePolicySAC('main', env, "JacoToss-v1", config)
	policy_vars = policy.get_variables()
	policy_path = load_model(path, policy_vars)
	return policy
