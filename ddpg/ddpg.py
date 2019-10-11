import numpy as np
import tensorflow as tf
import gym
import time

import ddpg.core as core
from ddpg.core import get_vars
from sac.utils.logx import EpochLogger

from ddpg.core import PrimitivePolicyDDPG
import baselines.common.tf_util as tf_util
from util import load_model, clip_reward, printstar

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
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
Deep Deterministic Policy Gradient (DDPG)
"""
def DDPG(env, path, config, primitives=None, bridge_policy=None,
        polyak=0.995, alpha=0.2, save_freq=1):
    """
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        pi_lr (float): Learning rate for policy.
        q_lr (float): Learning rate for Q-networks.
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    logger_kwargs = dict()
    logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())

    tf.set_random_seed(config.seed)
    np.random.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    main_policy = PrimitivePolicySAC('main', env, "JacoToss-v1", config)
    target_policy = PrimitivePolicySAC('target', env, "JacoToss-v1", config)

    # Inputs to computation graph
    x_ph, a_ph = main_policy.get_placeholders()
    x2_ph, _ = target_policy.get_placeholders()
    r_ph, d_ph = core.placeholders(None, None)

    # Main outputs from computation graph
    pi, q, q_pi = main_policy.get_actor_critic()
    
    # Target networks
    # Note that the action placeholder going to actor_critic here is 
    # irrelevant, because we only need q_targ(s, pi_targ(s)).
    pi_targ, _, q_pi_targ  = target_policy.get_actor_critic()

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=config.sac_replay_size)

    # Count variables
    var_counts = tuple(main_policy.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + config.gamma * (1 - d_ph) * q_pi_targ)

    # DDPG losses
    pi_loss = - tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q - backup) ** 2)

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=config.ddpg_pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=config.ddpg_q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess =  tf_util.get_session()
    var_list = main_policy.get_variables() + target_policy.get_variables() 
    ckpt_path = load_model(path, var_list)
    tf_util.initialize_vars(q_optimizer.variables())
    tf_util.initialize_vars(pi_optimizer.variables())

    # Setup model saving
    #logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

    def get_action(o, noise_scale):
        a, _ =  main_policy.step(o) #sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            while not(d or (ep_len == config.sac_max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        env.reset()

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = config.sac_steps_per_epoch * config.sac_epochs

    # NOTE : @dhruvramani
    num_primitives = len(primitives)
    curr_prim = 0
    if(config.stitch_naive):
        stitch_pi = primitives[curr_prim]
    else :
        stitch_pi = bridge_policy

    for t in range(total_steps):

        """
        Until config.ddpg_start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if(config.is_coart):
            a_p, q_p = stitch_pi.step(o)
            a = get_action(o, config.act_noise)
        else :
            if t > config.sac_start_steps:
                a = get_action(o, config.act_noise)
            else:
                a = env.action_space.sample()

        if(config.is_coart and not config.p1_value and config.stitch_naive and curr_prim == 0 and stitch_pi.is_terminate(o, init=True, env=env)): # and not config.imitate):
            curr_prim = 1
            stitch_pi = primitives[curr_prim]

        if config.render:
            env.render()
        
        if config.is_coart and (config.learn_higher_value or config.imitate):
            a = a_p

        # Step the env
        o2, r, d, _ = env.step(a)

        if config.is_coart and (config.learn_higher_value or config.p1_value):
            _, q_p1  = stitch_pi.step(o2)
            r = q_p1 - q_p
            r = clip_reward(r, lower_lim=0.0, upper_lim=25.0, scale=100.0)
            if(curr_prim == 1):
                r = r * config.ps_value_scale

            if config.debug:
                print("Prim : {} - Value : {}".format(curr_prim, r))

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
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(config.sac_batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }

                # Q-learning update
                outs = sess.run([q_loss, q, train_q_op], feed_dict)
                logger.store(LossQ=outs[0], QVals=outs[1])

                # Policy update
                outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            curr_prim = 0
            if(config.stitch_naive):
                stitch_pi = primitives[curr_prim]
            else :
                stitch_pi = bridge_policy

        # End of epoch wrap-up
        if t > 0 and t % config.sac_per_epoch == 0:
            epoch = t // config.sac_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == config.sac_epochs - 1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


            curr_prim = 0
            if(config.stitch_naive):
                stitch_pi = primitives[curr_prim]
            else :
                stitch_pi = bridge_policy

        return main_policy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
