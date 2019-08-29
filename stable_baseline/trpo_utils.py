import gym
import numpy as np
from collections import defaultdict

from stable_baselines.common.vec_env import VecEnv

# NOTE : @dhruvramani - refer to Stable-Baseline's for the original code

def traj_segment_generator_bridge(policy, primitives, env, horizon, config, reward_giver=None, gail=False):
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
        if(curr_prim == 0 and pi.is_terminate(ob, init=True, env=env)):
            curr_prim = 1
            if(config.stitch_naive):
                pi = primitives[curr_prim]
            else :
                pi = policy

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
        
        if(config.render):
            env.render()

        for key, value in info.items():
            reward_info[key].append(value)

        if(curr_prim == 1):
            rew = primitives[curr_prim].value(ob) - primitives[curr_prim].value(old_ob)

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

def traj_segment_generator(pi, env, horizon, config, reward_giver=None, gail=False):
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

        if(config.render):
            env.render()

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
