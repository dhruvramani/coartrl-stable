import numpy as np
import tensorflow as tf
import gym
import time
import sac.core as core
from sac.core import get_vars
from sac.utils.logx import EpochLogger
from sac.utils.run_utils import setup_logger_kwargs

from sac.core import PrimitivePolicySAC
import baselines.common.tf_util as tf_util
from util import load_model

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""
Soft Actor-Critic
(With slight variations that bring it closer to TD3)
"""
def SAC(env, test_env, path, config, primitives=None, bridge_policy=None,
        polyak=0.995, alpha=0.2, save_freq=1):
    
    logger_kwargs = setup_logger_kwargs("JacoSAC", 0)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(config.seed)
    np.random.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    main_policy = PrimitivePolicySAC('main', env, "JacoToss-v1", config)
    target_policy = PrimitivePolicySAC('target', env, "JacoToss-v1", config)
    
    x_ph, a_ph = main_policy.get_placeholders()
    x2_ph, _ = target_policy.get_placeholders()
    r_ph, d_ph = core.placeholders(None, None)
    
    mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = main_policy.get_actor_critic()
    
    _, _, _, _, _, _, _, v_targ  = target_policy.get_actor_critic()

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=config.sac_replay_size)

    var_counts = tuple(main_policy.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])

    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + config.gamma * (1 - d_ph) * v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=config.sac_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list = main_policy.get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=config.sac_lr)
    value_params = main_policy.get_vars('main/q') + main_policy.get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(main_policy.get_variables(), target_policy.get_variables())])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(main_policy.get_variables(), target_policy.get_variables())])
                              
    sess = tf_util.get_session()
    #sess.run(tf.global_variables_initializer())
    tf_util.initialize()
    var_list = main_policy.get_variables() + target_policy.get_variables()
    load_model(path, var_list)
    #tf_util.initialize_vars(var_list)
    #sess.run(target_init)

    # Setup model saving
    #logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def test_agent(n=10):
        global main_policy
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == config.sac_max_ep_len)):
                # Take deterministic actions at test time
                action, _ = main_policy.step(o, deterministic=True)
                o, r, d, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps, t = config.sac_steps_per_epoch * config.sac_epochs, 0

    # Main loop: collect experience in env and update/log each epoch
    while t in range(total_steps):

        """
        Until config.sac_start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > config.sac_start_steps:
            a = main_policy.step(o)
        else:
            a = env.action_space.sample()
        # Step the env
        # print('Action:',a)
        # print('#'*50)
        if config.render:
            env.render()
        
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == config.num_rollouts else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == config.num_rollouts):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(config.sac_batch_size)
                feed_dict = {a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                
                obs1_list = main_policy.get_ob_list(batch['obs1'])
                obs2_list = target_policy.get_ob_list(batch['obs2'])
                
                for i, x_i in enumerate(x_ph):
                    feed_dict[x_i] = obs1_list[i]

                for i, x_i in enumerate(x2_ph):
                    feed_dict[x_i] = obs2_list[i]

                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                             VVals=outs[6], LogPi=outs[7])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % config.sac_steps_per_epoch == 0:
            epoch = t // config.sac_steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == config.sac_epochs - 1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            print('TESTING AGENT\n\n')
            test_agent()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            logger.log_tabular('Q2Vals', with_min_and_max=True) 
            logger.log_tabular('VVals', with_min_and_max=True) 
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    return main_policy
