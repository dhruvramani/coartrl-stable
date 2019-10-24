import os.path as osp
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tqdm import trange
import h5py

from baselines.common import zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.statistics import stats

import rollouts as rollouts
from dataset import Dataset

class Trainer(object):
    def __init__(self, env, policy, old_policy, primitives, config, path):
        self._env = env
        self._path = path
        self._config = config
        self._num_primitives = len(primitives)

        self.policy = policy
        self.old_policy = old_policy
        self.primitives = primitives

        self._cur_lrmult = 0
        self._entcoeff = config.entcoeff
        self._optim_epochs = config.optim_epochs
        self._optim_stepsize = config.optim_stepsize
        self._optim_batchsize = config.optim_batchsize

        # global step
        self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)
        self.update_global_step = tf.assign(self.global_step, self.global_step + 1)

        # build loss/optimizers
        if self._config.coartl_method == 'trpo':
            self._build_trpo()
        elif self._config.coartl_method == 'ppo':
            self._build_ppo()
        else:
            raise NotImplementedError

        if self._is_chef and self._config.is_train:
            self.ep_stats = stats(self.summary_name)
            self.writer = U.file_writer(config.log_dir)

    @contextmanager
    def timed(self, msg):
        if self._is_chef:
            logger.info(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            logger.info(colorize("done in %.3f seconds" % (time.time()-tstart), color='magenta'))
        else:
            yield

    def _all_mean(self, x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= self._num_workers
        return out

    def _build_ppo(self):
        config = self._config
        pi = self.policy
        oldpi = self.old_policy

        # input placeholders
        obs = pi.obs
        ac = pi.pdtype.sample_placeholder([None], name='action')
        atarg = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage')
        ret = tf.placeholder(dtype=tf.float32, shape=[None], name='return')

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
        self._clip_param = config.clip_param * lrmult

        # policy
        var_list = pi.get_trainable_variables()
        self._adam = MpiAdam(var_list)

        fetch_dict = self.policy_loss_ppo(pi, oldpi, ac, atarg, ret)
        if self._is_chef:
            self.summary_name += ['ppo/' + key for key in fetch_dict.keys()]
            self.summary_name += ['ppo/grad_norm', 'ppo/grad_norm_clipped']
        fetch_dict['g'] = U.flatgrad(fetch_dict['total_loss'], var_list)
        self._loss = U.function([lrmult] + obs + [ac, atarg, ret], fetch_dict)
        self._update_oldpi = U.function([], [], updates=[
            tf.assign(oldv, newv) for (oldv, newv) in zipsame(
                oldpi.get_variables(), pi.get_variables())])

        # initialize and sync
        U.initialize()
        self._adam.sync()
   
    def _build_trpo(self):
        pi = self.policy
        oldpi = self.old_policy
        primitives = self.primitives

        # input placeholders
        obs = pi.obs
        ac = pi.pdtype.sample_placeholder([None], name='action')
        atarg = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage')  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None], name='return')  # Empirical return
        prim1 = tf.placeholder(dtype=tf.float32, shape=[None], name='prim1')

        # policy
        all_var_list = pi.get_trainable_variables()
        self.pol_var_list = [v for v in all_var_list if v.name.split("/")[2].startswith("pol")]
        self.vf_var_list = [v for v in all_var_list if v.name.split("/")[2].startswith("vf")]
        self._vf_adam = MpiAdam(self.vf_var_list)

        self.primitive_kl = self._config.kl_const * (prim1 * tf.reduce_mean(primitives[0].pd.kl(pi.pd)) + (1 - prim1) * tf.reduce_mean(primitives[1].pd.kl(pi.pd)))

        kl_oldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        mean_kl = tf.reduce_mean(kl_oldnew)
        mean_ent = tf.reduce_mean(ent)
        pol_entpen = - self._config.entcoeff * mean_ent

        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
        pol_surr = tf.reduce_mean(ratio * atarg)
        pol_loss = pol_surr + pol_entpen + self.primitive_kl

        pol_losses = {'pol_loss': pol_loss,
                      'pol_surr': pol_surr,
                      'pol_entpen': pol_entpen,
                      'kl': mean_kl,
                      'entropy': mean_ent, #}
                      'primitive_kl' : self.primitive_kl}

        if self._is_chef:
            self.summary_name += ['trpo/vf_loss']
            self.summary_name += ['trpo/' + key for key in pol_losses.keys()]

        self._get_flat = U.GetFlat(self.pol_var_list)
        self._set_from_flat = U.SetFromFlat(self.pol_var_list)
        klgrads = tf.gradients(mean_kl, self.pol_var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in self.pol_var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
        fvp = U.flatgrad(gvp, self.pol_var_list)

        self.compute_primitive_kl = U.function(obs + [ac, atarg, prim1], self.primitive_kl)
        self._update_oldpi = U.function([], [], updates=[
            tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        self._compute_losses = U.function(obs + [ac, atarg, prim1], pol_losses)
        pol_losses = dict(pol_losses)
        pol_losses.update({'g': U.flatgrad(pol_loss, self.pol_var_list)})
        self._compute_lossandgrad = U.function(obs + [ac, atarg, prim1], pol_losses)
        self._compute_fvp = U.function([flat_tangent] + obs + [ac, atarg], fvp)
        self._compute_vflossandgrad = U.function(obs + [ret], U.flatgrad(vf_loss, self.vf_var_list))
        self._compute_vfloss = U.function(obs + [ret], vf_loss)

        # initialize and sync
        U.initialize()
        th_init = self._get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        self._set_from_flat(th_init)
        self._vf_adam.sync()
        
    def policy_loss_ppo(self, pi, oldpi, ac, atarg, ret):
        kl_oldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        mean_kl = U.mean(kl_oldnew)
        mean_ent = U.mean(ent)
        pol_entpen = - self._entcoeff * mean_ent

        action_prob = pi.pd.logp(ac) - oldpi.pd.logp(ac)
        action_loss = tf.exp(action_prob) * atarg

        ratio = tf.exp(action_prob)

        surr1 = ratio * atarg
        surr2 = U.clip(ratio, 1.0 - self._clip_param, 1.0 + self._clip_param) * atarg
        pol_surr = -U.mean(tf.minimum(surr1, surr2))
        vf_loss = U.mean(tf.square(pi.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss

        losses = {'total_loss': total_loss,
                  'action_loss': action_loss,
                  'pol_surr': pol_surr,
                  'pol_entpen': pol_entpen,
                  'kl': mean_kl,
                  'entropy': mean_ent,
                  'vf_loss': vf_loss}

        return losses

    def _summary(self, it):
        if self._is_chef:
            if it % self._config.ckpt_save_step == 0:
                fname = osp.join(self._path, '%.5d' % it)
                U.save_state(fname)

    def train(self, rollout):
        config = self._config
        sess = U.get_session()
        global_step = sess.run(self.global_step)
        t = trange(global_step, config.max_iters,
                   total=config.max_iters, initial=global_step)
        info = None

        for step in t:
            # backup checkpoint
            self._summary(step)
            self._cur_lrmult = max(1.0 - float(step) / config.max_iters, 0)

            # rollout
            with self.timed("sampling"):
                rolls = rollout.__next__()
            if config.coartl_method == 'trpo':
                rollouts.add_advantage_rl(rolls, 0.99, 0.98)
            elif config.coartl_method == 'ppo':
                rollouts.add_advantage_rl(rolls, 0.99, 0.95)

            # train policy
            info = self._update_policy(rolls, step)
            if self._is_chef:
                ep = len(rolls["ep_length"])
                reward_mean = np.mean(rolls["ep_reward"])
                reward_std = np.std(rolls["ep_reward"])
                length_mean = np.mean(rolls["ep_length"])
                length_std = np.std(rolls["ep_length"])
                desc = "ep(%d) reward(%.1f, %.1f) length(%d, %.1f)" % (ep, reward_mean, reward_std, length_mean, length_std)
                for key, value in rolls.items():
                    if key.startswith('ep_'):
                        info[key.split('ep_')[1]] = np.mean(value)

            # log
            if self._is_chef:
                self.ep_stats.add_all_summary_dict(self.writer, info, global_step)
                t.set_description(desc)
                global_step = sess.run(self.update_global_step)

    def _update_policy(self, seg, it):
        if self._config.coartl_method == 'trpo':
            info = self._update_policy_trpo(seg, it)
        elif self._config.coartl_method == 'ppo':
            info = self._update_policy_ppo(seg)
        return info

    def _update_policy_trpo(self, seg, it):
        pi = self.policy
        ob, ac, atarg, tdlamret, prim1s = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["prim1s"]
        atarg = (atarg - atarg.mean()) / atarg.std()

        if self._is_chef:
            info = defaultdict(list)

        ob_dict = self._env.get_ob_dict(ob)
        for ob_name in pi.ob_type:
            pi.ob_rms[ob_name].update(ob_dict[ob_name])

        ob_list = pi.get_ob_list(ob_dict)
        args = ob_list + [ac, atarg, prim1s]
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return self._all_mean(self._compute_fvp(p, *fvpargs)) + self._config.cg_damping * p

        self._update_oldpi()

        with self.timed("computegrad"):
            lossbefore = self._compute_lossandgrad(*args)
            lossbefore = {k: self._all_mean(np.array(lossbefore[k])) for k in sorted(lossbefore.keys())}
        g = lossbefore['g']

        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with self.timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=self._config.cg_iters, verbose=self._is_chef)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self._config.max_kl)
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore['pol_loss']
            stepsize = 1.0
            thbefore = self._get_flat()
            for _ in range(10):

                primitive_klval = self.compute_primitive_kl(*args) # Treat is like logging
                with open('./klvalue.txt', 'w+') as f:
                    f.write("{}\n".format(primitive_klval))

                thnew = thbefore + fullstep * stepsize
                self._set_from_flat(thnew)
                meanlosses = self._compute_losses(*args)
                meanlosses = {k: self._all_mean(np.array(meanlosses[k])) for k in sorted(meanlosses.keys())}
                # logger.info('mean', [float(meanlosses[k]) for k in ['pol_loss', 'kl', 'pol_entpen', 'pol_surr', 'entropy']])
                if self._is_chef:
                    for key, value in meanlosses.items():
                        if key != 'g':
                            info['trpo/' + key].append(value)
                surr = meanlosses['pol_loss']
                kl = meanlosses['kl']
                meanlosses = np.array(list(meanlosses.values()))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > self._config.max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                self._set_from_flat(thbefore)
            if self._num_workers > 1 and it % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self._vf_adam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        with self.timed("vf"):
            for _ in range(self._config.vf_iters):
                for (mbob, mbret) in dataset.iterbatches(
                        (ob, tdlamret), include_final_partial_batch=False, batch_size=64):
                    ob_list = pi.get_ob_list(mbob)
                    g = self._all_mean(self._compute_vflossandgrad(*ob_list, mbret))
                    self._vf_adam.update(g, self._config.vf_stepsize)
                    vf_loss = self._all_mean(np.array(self._compute_vfloss(*ob_list, mbret)))
                    if self._is_chef:
                        info['trpo/vf_loss'].append(vf_loss)

        if self._is_chef:
            for key, value in info.items():
                info[key] = np.mean(value)
            return info
        return None

    def _update_policy_ppo(self, seg):
        pi = self.policy
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        atarg = (atarg - atarg.mean()) / max(atarg.std(), 0.000001)

        if self._is_chef:
            info = defaultdict(list)

        optim_batchsize = min(self._optim_batchsize, ob.shape[0])
        # prepare batches
        d = dataset.Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)

        ob_dict = self._env.get_ob_dict(ob)
        for ob_name in pi.ob_type:
            pi.ob_rms[ob_name].update(ob_dict[ob_name])

        self._update_oldpi()

        with self.timed("update"):
            for _ in range(self._optim_epochs):
                for batch in d.iterate_once(optim_batchsize):
                    ob_list = pi.get_ob_list(batch["ob"])
                    fetched = self._loss(self._cur_lrmult, *ob_list,
                                        batch["ac"], batch["atarg"], batch["vtarg"])
                    self._adam.update(fetched['g'], self._optim_stepsize * self._cur_lrmult)
                    if self._is_chef:
                        for key, value in fetched.items():
                            if key != 'g':
                                if np.isscalar(value):
                                    info['ppo/' + key].append(value)
                                else:
                                    info['ppo/' + key].extend(value)
                            else:
                                grad_norm_value = np.linalg.norm(value)
                                info['ppo/grad_norm'].append(grad_norm_value)
                                info['ppo/grad_norm_clipped'].append(np.clip(
                                    grad_norm_value, 0, self._config.trans_max_grad_norm))

        if self._is_chef:
            for key, value in info.items():
                info[key] = np.mean(value)
            return info
        return None


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

