import os
import logging
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as tf_util
from baselines.common.atari_wrappers import TransitionEnvWrapper
from baselines.common.mpi_running_mean_std import RunningMeanStd


def make_env(env_name, config=None):
    import gym
    env = gym.make(env_name)
    gym.logger.setLevel(logging.WARN)
    if config:
        assert env.spec.max_episode_steps <= config.total_timesteps, \
            '--num_rollouts ({}) should be larger than a game length ({})'.format(
                config.total_timesteps, env.spec.max_episode_steps)

    env = TransitionEnvWrapper(env)
    return env

def activation(activation):
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'elu':
        return tf.nn.elu
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.2)
    elif activation == 'tanh':
        return tf.tanh
    elif activation == 'sigmoid':
        return tf.sigmoid
    else:
        raise NotImplementedError('{} is not implemented'.format(activation))

def load_model(load_model_path, var_list=None):
    if os.path.isdir(load_model_path):
        ckpt_path = tf.train.latest_checkpoint(load_model_path)
    else:
        ckpt_path = load_model_path
        os.mkdir(ckpt_path)
        tf_util.initialize_vars(var_list)
        ckpt_path = ckpt_path + "/" + ckpt_path.split("/")[-1]
        tf_util.save_state(ckpt_path, var_list)
    if ckpt_path:
        tf_util.load_state(ckpt_path, var_list)
    return ckpt_path

    
class ObsNormalizer(object):
    def __init__(self, env, ob_env_name, config, num_policies=0, policy_const=None, policy_args=None):
        self.env = env
        self.config = config
        self.ob_env_name = ob_env_name
        self.name = "Obs_normal/{}".format(self.ob_env_name)
        self._include_acc = config.primitive_include_acc

        primitive_env = make_env(ob_env_name, config)
        self.hard_coded = primitive_env.hard_coded
        self._ob_shape = primitive_env.ob_shape
        self.ob_type = sorted(primitive_env.ob_type)

        if not self._include_acc and 'acc' in self.ob_type:
            self._ob_shape.pop('acc')
            self.ob_type.remove('acc')

        self._ob_space = np.sum([np.prod(ob) for ob in self._ob_shape.values()])

        with tf.variable_scope(self.name):
            self._obv_build()

        if(num_policies > 0):  
            self.policies = []
            for i in range(num_policies):
                self.policies.append(policy_const(*policy_args[:-1],  obs_phs=self.obs_phs, **policy_args[-1]))

    def _obv_build(self):
        self._obs = {}
        for ob_name, ob_shape in self._ob_shape.items():
            self._obs[ob_name] = tf_util.get_placeholder(
                name="ob_{}_primitive".format(ob_name),
                dtype=tf.float32,
                shape=[None] + self._ob_shape[ob_name])

        self.ob_rms = {}
        for ob_name in self.ob_type:
            with tf.variable_scope("ob_rms_{}".format(ob_name)):
                self.ob_rms[ob_name] = RunningMeanStd(shape=self._ob_shape[ob_name])
        obz = [(self._obs[ob_name] - self.ob_rms[ob_name].mean) / self.ob_rms[ob_name].std
               for ob_name in self.ob_type]
        obz = [tf.clip_by_value(ob, -5.0, 5.0) for ob in obz]
        self.obz = tf.concat(obz, -1)

        self.obs = [self._obs[ob_name] for ob_name in self.ob_type]
        self.obs = tf.concat(self.obs, -1)
        self.obs_phs = (self.obs, self.obz)

    def get_ob_dict(self, ob):
        if not isinstance(ob, dict):
            ob = self._env.get_ob_dict(ob)
        ob_dict = {}
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                t_ob = ob[ob_name][None]
            else:
                t_ob = ob[ob_name]
            t_ob = t_ob[:, -np.sum(self._ob_shape[ob_name]):]
            ob_dict[ob_name] = t_ob
        return ob_dict

    def get_ob_list(self, ob):
        ob_list = []
        if not isinstance(ob, dict):
            ob = self._env.get_ob_dict(ob)
        for ob_name in self.ob_type:
            if len(ob[ob_name].shape) == 1:
                t_ob = ob[ob_name][None]
            else:
                t_ob = ob[ob_name]
            t_ob = t_ob[:, -np.sum(self._ob_shape[ob_name]):]
            ob_list.append(t_ob)
        return ob_list

    def normalize_obv(self, env, obs):
        ob_dict = env.get_ob_dict(obs)
        for ob_name in self.ob_type:
            self.ob_rms[ob_name].update(ob_dict[ob_name])

        ob_list = self.get_ob_list(ob_dict)
        return ob_list
