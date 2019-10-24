import os
import sys
import statistics 
import numpy as np

from sac.sac import SAC
from sac.core import PrimitivePolicySAC

from transition.trainer import Trainer
from transition.ppolicy import PrimitivePolicy
from transition.rollouts import traj_segment_generator_coart

from util import *

def get_po_policy(env, primitives, higher_value, config, path):
	policy = PrimitivePolicy(env=env, name="%s/pi" % path, ob_env_name="JacoToss-v1", config=config)
	old_policy = PrimitivePolicy(env=env, name="%s/old_pi" % path, ob_env_name="JacoToss-v1", config=config)

	varlist = policy.get_variables() + old_policy.get_variables()
	policy_path = load_model(path, varlist)

	trainer = Trainer(env, policy, old_policy, primitives, config, path)
	rollout = traj_segment_generator_coart(env, policy, primitives, higher_value, stochastic=True, config=config)
	trainer.train(rollout)
	return policy

def get_higher_value(env, config, primitives):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.higher_value_path))
	if(os.path.exists(path)):
		printstar("Loading Higher Level Value Function")
		model = load_coartl(env, config, path, "sac")
	else :
		config.learn_higher_value = True
		model = SAC(env, path, config, primitives=primitives)
	return model


def get_coartl(env, config, primitives=None):
	model = None
	path = os.path.expanduser(os.path.join(config.policy_dir, config.coartl_path))
	higher_value = None
	#higher_value = get_higher_value(env, config, primitives)

	if(os.path.exists(path)):
		printstar("Loading {}".format(config.coartl_method.upper()))
		model = load_coartl(env, config, path, config.coartl_method) 
	
	if(config.is_train or model is None):
		config.is_train = True
		printstar("Training {}".format(config.coartl_method.upper()))
		
		if(config.coartl_method == 'trpo' or config.coartl_method == 'ppo'):
			model = get_po_policy(env, primitives, higher_value, config, path)
		elif(config.coartl_method == 'sac'):
			model = SAC(env, path, config, primitives=primitives)
		elif(config.coartl_method == 'ddpg'):
			model = DDPG(env, path, config, primitives=primitives)

	return model

def load_coartl(env, config, path, method):
	if(method == 'trpo' or method == 'ppo'):
		policy = PrimitivePolicy(env=env, name="%s/pi" % path, ob_env_name="JacoToss-v1", config=config)
	elif(method == 'sac'):
		policy = PrimitivePolicySAC('main', env, "JacoToss-v1", config)
	elif(method == 'ddpg'):
		policy = PrimitivePolicyDDPG('main', env, "JacoToss-v1", config)

	policy_vars = policy.get_variables()
	policy_path = load_model(path, policy_vars)
	return policy
