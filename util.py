import os
import logging
import tensorflow as tf
import baselines.common.tf_util as tf_util
from baselines.common.atari_wrappers import TransitionEnvWrapper

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
        #init = tf.global_variables_initializer()
        #sess.run(init)
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
