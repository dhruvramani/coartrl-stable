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
		trainer = TRPO(PrimitivePolicy, env, primitives=primitives, config=config, env_name=config.bridge_path, save_path=path)
		trainer.learn(total_timesteps=config.total_timesteps)
		model = trainer.policy_pi

	if(config.eval_all):
		evaluate_policy(env, model, config)
	return model

def get_coartl(env, config, primitives=None, bridge_policy=None):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.coartl_path))
	if(os.path.exists(path)):
		printstar("Loading {}".format(config.coartl_method.upper()))
		model = load_coartl(env, config, path) 
	
	if(config.is_train or model is None):
		printstar("Training {}".format(config.coartl_method.upper()))
		
		if(config.coartl_method == 'trpo'):
			trainer = TRPO(PrimitivePolicy, env, primitives=primitives[0], config=config, env_name=config.bridge_path, save_path=path)
			trainer.learn(total_timesteps=config.total_timesteps)
			model = trainer.policy_pi
		elif(config.coartl_method == 'sac'):
			model = SAC(env, path, config, primitives=primitives, bridge_policy=bridge_policy)
		elif(config.coartl_method == 'ddpg'):
			model = DDPG(env, path, config, primitives=primitives, bridge_policy=bridge_policy)

	if(config.eval_all):
		evaluate_policy(env, model, config)

	return model

def load_coartl(env, config, path):
	if(config.coartl_method == 'trpo'):
		policy = PrimitivePolicy(env=env, name="%s/pi" % env_name, ob_env_name="JacoToss-v1", config=config, n_env=1)
	elif(config.coartl_method == 'sac'):
		policy = PrimitivePolicySAC('main', env, "JacoToss-v1", config)
	elif(config.coartl_method == 'ddpg'):
		policy = PrimitivePolicyDDPG('main', env, "JacoToss-v1", config)

	policy_vars = policy.get_variables()
	policy_path = load_model(path, policy_vars)
	return policy
