import gym
import numpy as np
from collections import defaultdict

from stable_baselines.common.vec_env import VecEnv

'''
def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:

        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0 # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    states = policy.initial_state
    episode_start = True  # marks if we're on first timestep of an episode
    done = False

    while True:
        action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len
            }
            _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred
        actions[i] = action[0]
        episode_starts[i] = episode_start

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if gail:
            reward = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_reward, done, info = env.step(clipped_action[0])
        else:
            observation, reward, done, info = env.step(clipped_action[0])
            true_reward = reward
        rewards[i] = reward
        true_rewards[i] = true_reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        cur_ep_true_ret += true_reward
        current_it_len += 1
        current_ep_len += 1
        if done:
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                if not gail:
                    cur_ep_ret = maybe_ep_info['r']
                cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1
'''

def traj_segment_generator_bridge(primitives, env, horizon, reward_giver=None, gail=False):
    t = 0
    ac = env.action_space.sample()
    done = False
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    ep_reward = defaultdict(list)

    # Initialize history arrays
    obs = []
    acs = []
    vpreds = []
    rews = []
    dones = []
    reward_info = defaultdict(list)

    num_primitives = len(primitives)
    curr_prim = 0
    pi = primitives[curr_prim] # NOTE : @dhruvramani - pi depicts the current primitive

    while True:
        if(pi.is_terminate(ob, init=True, env=env)):
            curr_prim += 1
            curr_prim = curr_prim % num_primitives
            pi = primitives[curr_prim]
            print("Changed Policy")

        ac, vpred, _, _ = pi.step(ob)

        if t > 0 and t % horizon == 0:
            dicti = {"observations": obs, "rewards": rews, "vpred": vpreds, "nextvpred": vpred * (1 - done),
                     "dones": dones, "actions": acs, "ep_rets": ep_rets, "ep_lens": ep_lens, "total_timestep": t, }
            for key, value in ep_reward.items():
                dicti.update({"ep_{}".format(key): value})
            yield {key: np.copy(val) for key, val in dicti.items()}
            ep_rets = []
            ep_lens = []
            ep_reward = defaultdict(list)
            obs = []
            rews = []
            vpreds = []
            dones = []
            acs = []
            t = 0
            env.reset()
            curr_prim = 0
            pi = primitives[curr_prim] 
            vpred = pi.value(ob)
        obs.append(ob)
        vpreds.append(vpred)
        acs.append(ac)

        old_ob = ob
        ob, rew, done, info = env.step(ac)
        env.render()
        for key, value in info.items():
            reward_info[key].append(value)

        if(curr_prim != 0):
            rew = pi.value(ob) - pi.value(old_ob)

        rews.append(rew)
        dones.append(done)
        cur_ep_ret += rew
        cur_ep_len += 1
        t += 1

        if done:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            curr_prim = 0
            pi = primitives[curr_prim] 
            for key, value in reward_info.items():
                if isinstance(value[0], (int, float, np.bool_)):
                    if '_mean' in key:
                        ep_reward[key].append(np.mean(value))
                    else:
                        ep_reward[key].append(np.sum(value))

def traj_segment_generator(pi, env, horizon, reward_giver=None, gail=False):
    t = 0
    ac = env.action_space.sample()
    done = False
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    ep_reward = defaultdict(list)

    # Initialize history arrays
    obs = []
    acs = []
    vpreds = []
    rews = []
    dones = []
    reward_info = defaultdict(list)

    while True:
        ac, vpred, _, _ = pi.step(ob)

        if t > 0 and t % horizon == 0:
            dicti = {"observations": obs, "rewards": rews, "vpred": vpreds, "nextvpred": vpred * (1 - done),
                     "dones": dones, "actions": acs, "ep_rets": ep_rets, "ep_lens": ep_lens, "total_timestep": t, }
            for key, value in ep_reward.items():
                dicti.update({"ep_{}".format(key): value})
            yield {key: np.copy(val) for key, val in dicti.items()}
            ep_rets = []
            ep_lens = []
            ep_reward = defaultdict(list)
            obs = []
            rews = []
            vpreds = []
            dones = []
            acs = []
            t = 0
            vpred = pi.value(ob)
        obs.append(ob)
        vpreds.append(vpred)
        acs.append(ac)

        ob, rew, done, info = env.step(ac)
        for key, value in info.items():
            reward_info[key].append(value)
        rews.append(rew)
        dones.append(done)
        cur_ep_ret += rew
        cur_ep_len += 1
        t += 1

        if done:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            for key, value in reward_info.items():
                if isinstance(value[0], (int, float, np.bool_)):
                    if '_mean' in key:
                        ep_reward[key].append(np.mean(value))
                    else:
                        ep_reward[key].append(np.sum(value))

'''
def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]
'''
def add_vtarg_and_adv(seg, gamma, lam):
    # print(seg.keys())
    done = seg["dones"]
    rew = seg["rewards"]
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(rew)
    seg["adv"] = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - done[t]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        gaelam[t] = lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

    assert np.isfinite(seg["vpred"]).all()
    assert np.isfinite(seg["nextvpred"]).all()
    assert np.isfinite(seg["adv"]).all()


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]
