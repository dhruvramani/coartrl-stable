import tensorflow as tf
import numpy as np
import gym

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.common.distributions import CategoricalPdType
from baselines.common.atari_wrappers import TransitionEnvWrapper
import baselines.common.tf_util as U

from util import make_env, activation


EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    # vfs
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('v'):
        v = vf_mlp(x)
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v


class PrimitivePolicySAC(object):
    def __init__(self, name, env, prim_env_name, config):
        # configs
        self._config = config

        # args
        self.name = name
        self.env_name = self.name.split('-')[0]

        # training
        self._hid_size = config.sac_hid_size 
        self._num_hid_layers = config.sac_num_hid_layers
        self._activation = activation(config.sac_activation)
        self._include_acc = config.primitive_include_acc

        # properties
        self.prim_env_name = prim_env_name
        primitive_env = make_env(prim_env_name, config)
        self.hard_coded = primitive_env.hard_coded
        self._ob_shape = primitive_env.ob_shape
        self.ob_type = sorted(primitive_env.ob_type)

        if not self._include_acc and 'acc' in self.ob_type:
            self._ob_shape.pop('acc')
            self.ob_type.remove('acc')

        self._env = env
        self._ob_space = np.sum([np.prod(ob) for ob in self._ob_shape.values()])
        self._ac_space = primitive_env.action_space
        self.act_dim = self._ac_space.shape[0]

        if config.primitive_use_term:
            self.primitive_env = primitive_env
        else:
            primitive_env.close()

        with tf.variable_scope(self.name):
            self._ph_build()
        
        if not self.hard_coded:
            with tf.variable_scope(self.name):
                self._scope = tf.get_variable_scope().name
                self._build()

    def _ph_build(self):
        self._obs = {}
        self.action_ph = placeholder(self.act_dim)

        for ob_name, ob_shape in self._ob_shape.items():
            self._obs[ob_name] = U.get_placeholder(
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

    def _build(self):
        ac_space = self._ac_space
        hid_size = self._hid_size
        num_hid_layers = self._num_hid_layers

        # primitive policy
        self.pdtype = pdtype = make_pdtype(ac_space)
        # NOTE : @dhruvramani - modifying code
        with tf.variable_scope("pi"):
            net = self.obz
            for i in range(num_hid_layers):
                net = self._activation(tf.layers.dense(net, hid_size, name="fc%i" % (i+1), kernel_initializer=U.normc_initializer(1.0)))

            mu = tf.layers.dense(net, self.act_dim, activation=None)
            log_std = tf.layers.dense(net, self.act_dim, activation=tf.tanh)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp_pi = gaussian_likelihood(pi, mu, log_std)

            mu = tf.tanh(mu)
            pi = tf.tanh(pi)
            logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)

            action_scale = ac_space.high[0]
            mu *= action_scale
            pi *= action_scale

            self.mu, self.pi, self.logp_pi = mu, pi, logp_pi
            
            # if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            #     mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name="final", kernel_initializer=U.normc_initializer(0.01))
            #     logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            #     pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            # else:
            #     pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name="final", kernel_initializer=U.normc_initializer(0.01))

        # value function
        def value_mvp(scope_name, input, reuse=False):
            with tf.variable_scope(scope_name, reuse=reuse):
                net = input
                for i in range(num_hid_layers):
                    net = self._activation(tf.layers.dense(net, hid_size, name="fc%i" % (i+1), kernel_initializer=U.normc_initializer(1.0)))
                
                vpred = tf.squeeze(tf.layers.dense(net, 1, name="final", kernel_initializer=U.normc_initializer(1.0)), axis=1)
            return vpred

        q1 = value_mvp("q1", tf.concat([self.obz, self.action_ph], axis=-1))
        q1_pi = value_mvp("q1", tf.concat([self.obz, self.pi], axis=-1), reuse=True)
        q2 = value_mvp("q2", tf.concat([self.obz, self.action_ph], axis=-1))
        q2_pi = value_mvp("q2", tf.concat([self.obz, self.pi], axis=-1), reuse=True)        
        v = value_mvp("v", self.obz)

        self.q1, self.q1_pi, self.q2, self.q2_pi, self.v = q1, q1_pi, q2, q2_pi, v


        #self.pd = pdtype.proba_distribution_from_flat(pdparam)
        # sample action
        #stochastic = tf.placeholder(dtype=tf.bool, shape=())
        #ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function(self.obs, [self.mu, self.pi, self.v])
        self._value = U.function(self.obs, [self.q1, self.q1_pi, self.q2, self.q2_pi, self.v])

    def step(self, obs, deterministic=False):
        stochastic = not deterministic
        if self.hard_coded:
            return self.primitive_env.unwrapped.act(osb), 0
        ob_list = self.get_ob_list(obs)
        mu, pi, vpred = self._act(*ob_list)
        ac = mu if deterministic else pi
        return ac[0], vpred[0]

    def value(self, obs, deterministic=False):
        stochastic = not deterministic
        if self.hard_coded:
            return 0
        ob_list = self.get_ob_list(obs)
        q1, q1_pi, q2, q2_pi, v = self._value(*ob_list)
        return q1[0], q1_pi[0], q2[0], q2_pi[0], v[0]

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

    def is_terminate(self, ob, init=False, env=None):
        return self.primitive_env.unwrapped.is_terminate(ob, init=init, env=env)

    def get_placeholders(self):
        return self.obs, self.action_ph

    def get_actor_critic(self):
        return self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v

    def get_variables(self):
        if self.hard_coded:
            return []
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope)

    def get_trainable_variables(self):
        if self.hard_coded:
            return []
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)
        return var_list

    def get_vars(self, scope):
        return [x for x in self.get_variables() if scope in x.name]

    def count_vars(self, scope):
        v = get_vars(scope)
        return sum([np.prod(var.shape.as_list()) for var in v])

    def reset(self):
        if self.hard_coded:
            return
        with tf.variable_scope(self._scope, reuse=True):
            varlist = self.get_trainable_variables()
            initializer = tf.variables_initializer(varlist)
            U.get_session().run(initializer)
