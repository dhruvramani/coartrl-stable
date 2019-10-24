from collections import defaultdict
import os.path as osp
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from util import *


def render_frame(env, length, ret, primitive_name, render, record=False, caption_off=False):
    '''
    if not render and not record:
        return None
    raw_img = env.unwrapped.render_frame()
    raw_img = np.asarray(raw_img, dtype=np.uint8).copy()

    # write info
    if not caption_off:
        text = ['{:4d} {:.2f} {}'.format(length, ret, primitive_name)]
        x0, y0, dy = 10, 50, 50
        for i, t in enumerate(text):
            cv2.putText(raw_img, t, (x0, y0+i*dy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 128, 128), 2, cv2.LINE_AA)
    '''
    if render:
        env.render()
    '''
        import time
        time.sleep(0.03)
        raw_img = cv2.resize(raw_img, (500, 500))
        cv2.imshow(env.spec.id, raw_img)
        cv2.waitKey(1)
    return raw_img if record else None
    '''


class Rollout(object):
    def __init__(self):
        self._history = defaultdict(list)

    def clear(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        return self._history

def clip_reward(r, l=0., u=10.):
    if (r < l):
        return -1
    else:
        r = r / 100
        if (r < u):
            return r
        else:
            return u

def add_advantage_rl(seg, gamma, lam):
    # print(seg.keys())
    done = seg["done"]
    rew = seg["rew"]
    vpred = np.append(seg["vpred"], seg["next_vpred"])
    T = len(rew)
    seg["adv"] = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - done[t]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        # print(lastgaelam)
        gaelam[t] = lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

    assert np.isfinite(seg["vpred"]).all()
    assert np.isfinite(seg["next_vpred"]).all()
    assert np.isfinite(seg["adv"]).all()

def traj_segment_generator_coart(env, pi, primitives, higher_value, stochastic, config, training_inference=False):
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
    visual_obs = []
    acs = []
    vpreds = []
    rews = []
    prim1s = []
    dones = []
    reward_info = defaultdict(list)

    curr_prim = 0
    stitch_pi = primitives[0]

    if config.render:
        cv2.namedWindow(env.spec.id)
        cv2.moveWindow(env.spec.id, 0, 0)

    while True:
        #if(config.stitch_naive and curr_prim == 0 and stitch_pi.is_terminate(ob, init=True, env=env)):
        #    curr_prim = 1
        #    stitch_pi = primitives[curr_prim]

        ac, vpred = pi.step(ob, stochastic)

        if t >= config.num_rollouts and config.is_train and not training_inference:
            dicti = {"ob": obs, "rew": rews, "prim1s":prim1s, "vpred": vpreds, "next_vpred": vpred * (1 - done),
                     "done": dones, "ac": acs, "ep_reward": ep_rets, "ep_length": ep_lens}
            for key, value in ep_reward.items():
                dicti.update({"ep_{}".format(key): value})
            yield {key: np.copy(val) for key, val in dicti.items()}
            ep_rets = []
            ep_lens = []
            ep_reward = defaultdict(list)
            obs = []
            rews = []
            vpreds = []
            prim1s = []
            dones = []
            acs = []
            t = 0
            vpred = pi.value(ob, stochastic)

            curr_prim = 0
            stitch_pi = primitives[curr_prim]

        obs.append(ob)
        vpreds.append(vpred)
        
        vpred_p = stitch_pi.value(ob, stochastic) # higher_value.value(ob)
        
        acs.append(ac)
        vob = render_frame(env, cur_ep_len, cur_ep_ret, config.coartl_method, config.render, None, None)
        visual_obs.append(vob)
        
        ob, rew, done, info = env.step(ac)
        for key, value in info.items():
            reward_info[key].append(value)

        vpred_p1 = stitch_pi.value(ob, stochastic) #higher_value.value(ob) 

        rew = vpred_p1 - vpred_p 
        rew = clip_reward(rew)
        if(curr_prim == 1):
            rew = rew * config.ps_value_scale

        prim1 = float(curr_prim == 0)
        prim1s.append(prim1)
        rews.append(rew)
        dones.append(done)
        cur_ep_ret += rew
        cur_ep_len += 1
        t += 1

        if done:
            vob = render_frame(env, cur_ep_len, cur_ep_ret, config.coartl_method, config.render, None, None)
            visual_obs.append(vob)  # add last frame
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            curr_prim = 0
            stitch_pi = primitives[curr_prim]
            for key, value in reward_info.items():
                if isinstance(value[0], (int, float, np.bool_)):
                    if '_mean' in key:
                        ep_reward[key].append(np.mean(value))
                    else:
                        ep_reward[key].append(np.sum(value))

            if not config.is_train or training_inference:
                dicti = {"ep_reward": ep_rets, "ep_length": ep_lens, "visual_obs": visual_obs}
                if config.is_collect_state:
                    dicti["obs"] = obs
                for key, value in ep_reward.items():
                    dicti.update({"ep_{}".format(key): value})
                yield {key: np.copy(val) for key, val in dicti.items()}
                ep_rets = []
                ep_lens = []
                ep_reward = defaultdict(list)
                obs = []
                rews = []
                prim1s = []
                vpreds = []
                dones = []
                acs = []
                t = 0
            reward_info = defaultdict(list)
            cur_ep_ret = 0
            cur_ep_len = 0
            visual_obs = []
            ob = env.reset()

def traj_segment_generator_rl(env, pi, stochastic, config, training_inference=False):
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
    visual_obs = []
    acs = []
    vpreds = []
    rews = []
    dones = []
    reward_info = defaultdict(list)

    if config.render:
        cv2.namedWindow(env.spec.id)
        cv2.moveWindow(env.spec.id, 0, 0)

    while True:
        ac, vpred = pi.act(ob, stochastic)

        if t >= config.num_rollouts and config.is_train and not training_inference:
            dicti = {"ob": obs, "rew": rews, "vpred": vpreds, "next_vpred": vpred * (1 - done),
                     "done": dones, "ac": acs, "ep_reward": ep_rets, "ep_length": ep_lens}
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
            vpred = pi.value(ob, stochastic)
        obs.append(ob)
        vpreds.append(vpred)
        acs.append(ac)
        vob = render_frame(
            env, cur_ep_len, cur_ep_ret, config.rl_method, config.render,
            config.record, caption_off=config.video_caption_off)
        visual_obs.append(vob)

        ob, rew, done, info = env.step(ac)
        for key, value in info.items():
            reward_info[key].append(value)
        rews.append(rew)
        dones.append(done)
        cur_ep_ret += rew
        cur_ep_len += 1
        t += 1

        if done:
            vob = render_frame(
                env, cur_ep_len, cur_ep_ret, config.rl_method, config.render,
                config.record, caption_off=config.video_caption_off)
            visual_obs.append(vob)  # add last frame
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            for key, value in reward_info.items():
                if isinstance(value[0], (int, float, np.bool_)):
                    if '_mean' in key:
                        ep_reward[key].append(np.mean(value))
                    else:
                        ep_reward[key].append(np.sum(value))

            if not config.is_train or training_inference:
                dicti = {"ep_reward": ep_rets, "ep_length": ep_lens, "visual_obs": visual_obs}
                if config.is_collect_state:
                    dicti["obs"] = obs
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
            reward_info = defaultdict(list)
            cur_ep_ret = 0
            cur_ep_len = 0
            visual_obs = []
            if(config.num_rollouts > 1):
                ob = env.reset()


def add_advantage_meta(seg, gamma, lam, meta_duration):
    done = seg["meta_done"]
    rew = seg["meta_rew"]
    vpred = np.append(seg["meta_vpred"], 0)
    T = len(rew)
    seg["meta_adv"] = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - done[t]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["meta_tdlamret"] = seg["meta_adv"] + seg["meta_vpred"]
    assert np.isfinite(seg["meta_vpred"]).all()
    assert np.isfinite(seg["meta_adv"]).all()

def prepare_all_rolls(seg, gamma, lam, num_primitives, use_trans,
                      trans_use_task_reward, config):
    done = seg["done"]
    rew = seg["rew"]
    rew_task = seg["env_rew"]
    is_trans = np.append(seg["is_trans"], 0)
    vpred = np.append(seg["vpred"], 0)
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    ret = 0

    # mark successful trajectories
    success = False
    for t in reversed(range(T)):
        nonterminal = 1 - done[t]
        seg["success"][t] = success = (success and nonterminal) or seg["success"][t]

    # mark successful transitions
    if not config.use_trans_between_same_policy:
        success = True
        cur_primitive = -1
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            if nonterminal and not success and seg["cur_primitive"][t] == cur_primitive:
                seg["success"][t] = False
            cur_primitive = seg["cur_primitive"][t]
            success = seg["success"][t]

    for t in reversed(range(T)):
        nonterminal = 1 - done[t]
        # delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        if is_trans[t + 1]:
            ret = rew[t] + gamma * vpred[t + 1] * nonterminal
        else:
            # trans rew
            if trans_use_task_reward:
                ret = rew[t] + gamma * ret * nonterminal
            else:
                # use proximity_predictor
                ret = rew[t]

        delta = ret - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        if not is_trans[t]:
            lastgaelam = 0
            if not trans_use_task_reward:
                # separate trans within the same ep
                ret = 0

    seg["tdlamret"] = seg["adv"] + seg["vpred"]

    assert np.isfinite(seg["vpred"]).all()
    assert np.isfinite(seg["adv"]).all()

    split_test = split_segments(seg, num_primitives, use_trans)
    return split_test


def split_segments(seg, num_primitives, use_trans):
    trans_counts = [0] * num_primitives
    for cur_primitive, is_trans in zip(seg["cur_primitive"], seg["is_trans"]):
        if cur_primitive == num_primitives:
            continue
        if is_trans:
            trans_counts[cur_primitive] += 1

    trans = []
    if use_trans:
        for i in range(num_primitives):
            obs = np.array([seg["ob"][0] for _ in range(trans_counts[i])])
            cur_primitives = np.zeros(trans_counts[i], 'int32')
            prev_primitives = np.zeros(trans_counts[i], 'int32')
            advs = np.zeros(trans_counts[i], 'float32')
            tdlams = np.zeros(trans_counts[i], 'float32')
            terms = np.zeros(trans_counts[i], 'float32')
            success = np.zeros(trans_counts[i], 'float32')
            acs = np.array([seg["ac"][0] for _ in range(trans_counts[i])])
            trans.append({"ob": obs, "adv": advs, "tdlamret": tdlams, "ac": acs,
                          "cur_primitive": cur_primitives, "prev_primitive": prev_primitives,
                          "term": terms, "success": success})

    trans_counts = [0] * num_primitives
    for i in range(len(seg["ob"])):
        cur = seg["cur_primitive"][i]
        if cur == num_primitives:
            continue
        if use_trans and seg["is_trans"][i]:
            trans[cur]["ob"][trans_counts[cur]] = seg["ob"][i]
            trans[cur]["adv"][trans_counts[cur]] = seg["adv"][i]
            trans[cur]["tdlamret"][trans_counts[cur]] = seg["tdlamret"][i]
            trans[cur]["ac"][trans_counts[cur]] = seg["ac"][i]
            trans[cur]["cur_primitive"][trans_counts[cur]] = seg["cur_primitive"][i]
            trans[cur]["prev_primitive"][trans_counts[cur]] = seg["prev_primitive"][i]
            trans[cur]["term"][trans_counts[cur]] = seg["term"][i]
            trans[cur]["success"][trans_counts[cur]] = seg["success"][i]
            trans_counts[cur] += 1
    return trans
