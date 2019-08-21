import logging
import tensorflow as tf
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
