import numpy as np
import tensorflow as tf


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

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, q, q_pi

class PrimitivePolicyDDPG(object):
    def __init__(self, name, env, prim_env_name, config):
        # configs
        self._config = config

        # args
        self.name = name
        self.env_name = self.name.split('-')[0]

        # training
        self._hid_size = [int(i) for i in config.ddpg_hid_size] 
        self._num_hid_layers = config.ddpg_num_hid_layers
        self._activation = activation(config.ddpg_activation)
        self._include_acc = config.primitive_include_acc
        self._op_activation = config.ddpg_op_activation

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
        action_scale = ac_space.high[0]


        self.pdtype = pdtype = make_pdtype(ac_space)
        # NOTE : @dhruvramani - modifying code
        with tf.variable_scope("pi"):
            net = self.obz
            for i in range(num_hid_layers):
                net = self._activation(tf.layers.dense(net, hid_size[i], name="fc%i" % (i+1), kernel_initializer=U.normc_initializer(1.0)))

            pi = tf.layers.dense(net, self.act_dim, activation=self._op_activation)
            pi *= action_scale

        with tf.variable_scope("q"):
            qnet = tf.concat([self.obz, self.action_ph], axis=-1)
            for i in range(num_hid_layers):
                qnet = self._activation(tf.layers.dense(qnet, hid_size[i], name="fc%i" % (i+1)))
            q = tf.squeeze(tf.layers.dense(qnet, 1, activation=None), axis=1)

        with tf.variable_scope("q"):
            qnetp = tf.concat([self.obz, pi], axis=-1)
            for i in range(num_hid_layers):
                qnetp = self._activation(tf.layers.dense(qnetp, hid_size[i], name="fc%i" % (i+1)))
            q_pi = tf.squeeze(tf.layers.dense(qnetp, 1, activation=None), axis=1)

        self.pi, self.q, self.q_pi = pi, q, q_pi
        
        self._act = U.function(self.obs, [self.pi, self.q_pi])
        self._value = U.function(self.obs, [self.q_pi])

    def step(self, obs):
        if self.hard_coded:
            return self.primitive_env.unwrapped.act(osb), 0
        ob_list = self.get_ob_list(obs)
        pi, q_pi = self._act(*ob_list)
        return pi[0], q_pi[0]

    def value(self, obs):
        if self.hard_coded:
            return 0
        ob_list = self.get_ob_list(obs)
        q_pi = self._value(*ob_list)
        return q_pi[0]

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
        return self.pi, self.q, self.q_pi

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
