import time
import math
from copy import copy, deepcopy
import os
import os.path as osp
from collections import defaultdict, deque
import signal

import wandb
import numpy as np

from helpers import logger
# from helpers.distributed_util import sync_check
from helpers.console_util import timed_cm_wrapper, log_iter_info
from helpers.opencv_util import record_video


MAXLEN = 40

BRAC_BEHAV_STEPS = 2_000


def rollout_generator(env, agent, rollout_len, use_noise_process):

    t = 0
    if use_noise_process:
        # Reset agent's noise process
        agent.reset_noise()
    # Reset agent's env
    ob = np.array(env.reset())

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=True, max_q=None)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:
            yield

        # Interact with env(s)
        new_ob, rew, done, _ = env.step(ac)

        # Assemble and store transition in memory
        transition = {
            "obs0": ob,
            "acs": ac,
            "obs1": new_ob,
            "rews": rew,
            "dones1": done,
        }
        agent.store_transition(transition)

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            if use_noise_process:
                # Reset agent's noise process
                agent.reset_noise()
            # Reset agent's env
            ob = np.array(env.reset())

        t += 1


def ep_generator(env, agent, render, record, max_q):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    kwargs = {'mode': 'rgb_array'}

    def _render():
        return env.render(**kwargs)

    ob = np.array(env.reset())

    if record:
        ob_orig = _render()

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    if record:
        obs_render = []
    acs = []
    env_rews = []

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=False, max_q=max_q)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        obs.append(ob)
        if record:
            obs_render.append(ob_orig)
        acs.append(ac)
        new_ob, env_rew, done, _ = env.step(ac)

        if render:
            env.render()

        if record:
            ob_orig = _render()

        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = np.array(deepcopy(new_ob))
        if done:
            obs = np.array(obs)
            if record:
                obs_render = np.array(obs_render)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {"obs": obs,
                   "acs": acs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            if record:
                out.update({"obs_render": obs_render})

            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            if record:
                obs_render = []
            acs = []
            env_rews = []
            ob = np.array(env.reset())

            if record:
                ob_orig = _render()


def evaluate(args,
             env,
             agent_wrapper,
             experiment_name):

    # Create an agent
    agent = agent_wrapper()

    # Create episode generator
    ep_gen = ep_generator(env, agent, args.render)

    if args.record:
        vid_dir = osp.join(args.video_dir, experiment_name)
        os.makedirs(vid_dir, exist_ok=True)

    # Load the model
    agent.load(args.model_path, args.iter_num)
    logger.info("model loaded from path:\n  {}".format(args.model_path))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(args.num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, args.num_trajs))
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
        if args.record:
            # Record a video of the episode
            record_video(vid_dir, i, traj['obs_render'])

    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def generate_buffer(args,
                    env,
                    agent_wrapper,
                    experiment_name):

    # Set up where to save the exploration data
    explo_dir = osp.join(args.exploration_dir, experiment_name)
    os.makedirs(explo_dir, exist_ok=True)

    # Create an agent
    agent = agent_wrapper()

    # Log the type of exploration used by the agent
    logger.info("parameter noise used: {}".format(agent.param_noise))
    logger.info("action noise used: {}".format(agent.ac_noise))

    # Create episode generator
    roll_gen = rollout_generator(env, agent, rollout_len=args.gen_buffer_size)
    # Note, we only need to generate once from the generator (no training).

    # Load the model
    agent.load(args.model_path, args.iter_num)
    logger.info("model loaded from path:\n  {}".format(args.model_path))

    # Generate rollout to populate the replay buffer
    roll_gen.__next__()  # no need to get the returned rollout, stored in buffer

    print(agent.replay_buffer.num_entries)

    # Save the data on disk
    agent.save_full_history(explo_dir, "0-generated")  # technically no training is done here
    # Note, we use the full history method to be sure nothing is flushed


def learn(args,
          rank,
          env,
          main_eval_env,
          maxq_eval_env,
          agent_wrapper,
          experiment_name,
          use_noise_process=None):

    # Create an agent
    agent = agent_wrapper()

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    # Start clocks
    num_iters = int(args.num_steps) if args.offline else int(args.num_steps) // args.rollout_len
    iters_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()

    if rank == 0:
        # Create collections
        d = defaultdict(list)
        main_eval_deque = deque(maxlen=MAXLEN)
        maxq_eval_deque = deque(maxlen=MAXLEN)

        # Set up model save directory
        ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if args.record:
            vid_dir = osp.join(args.video_dir, experiment_name)
            os.makedirs(vid_dir, exist_ok=True)
        if not agent.hps.offline:
            # Set up replay save directory
            replay_dir = osp.join(args.replay_dir, experiment_name)
            os.makedirs(replay_dir, exist_ok=True)

        # Handle timeout signal gracefully
        def timeout(signum, frame):
            # Save the model
            agent.save(ckpt_dir, "{}_timeout".format(iters_so_far))
            # No need to log a message, orterun stopped the trace already
            # No need to end the run by hand, SIGKILL is sent by orterun fast enough after SIGTERM

        # Tie the timeout handler with the termination signal
        # Note, orterun relays SIGTERM and SIGINT to the workers as SIGTERM signals,
        # quickly followed by a SIGKILL signal (Open-MPI impl)
        signal.signal(signal.SIGTERM, timeout)

        # Group by everything except the seed, which is last, hence index -1
        group = '.'.join(experiment_name.split('.')[:-1])

        # Set up wandb
        while True:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=experiment_name,
                    id=experiment_name,
                    group=group,
                    config=args.__dict__,
                    dir=args.root,
                )
            except ConnectionRefusedError:
                pause = 5
                logger.info("wandb co error. Retrying in {} secs.".format(pause))
                time.sleep(pause)
                continue
            logger.info("wandb co established!")
            break

    # Create rollout generator for training the agent
    if args.offline:
        pass
    else:
        roll_gen = rollout_generator(env, agent, args.rollout_len, use_noise_process)
    if rank == 0:
        # Create episode generator for evaluating the agent
        main_eval_ep_gen = ep_generator(main_eval_env, agent, args.render, args.record, max_q=False)
        maxq_eval_ep_gen = ep_generator(maxq_eval_env, agent, args.render, args.record, max_q=True)

    while iters_so_far <= num_iters:

        log_iter_info(logger, iters_so_far, num_iters, tstart)

        # if iters_so_far % 20 == 0:
        #     # Check if the mpi workers are still synced
        #     sync_check(agent.actr)

        if rank == 0 and iters_so_far % args.save_frequency == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("time to checkpoint. Saving model @: {}".format(ckpt_dir))
            if not agent.hps.offline:
                # Save the replay buffer and the full history
                agent.save_memory(replay_dir, iters_so_far)
                agent.save_full_history(replay_dir, iters_so_far)

        if args.offline:
            pass
        else:
            # Sample mini-batch in env with perturbed actor and store transitions
            with timed("interacting"):
                roll_gen.__next__()  # no need to get the returned rollout, stored in buffer

        if args.algo.split('_')[0] == 'brac' and iters_so_far == 0:
            with timed('behavior policy pre-training'):
                # Train the behavior policy estimate in a pre-training phase, as in the BRAC codebase
                for i in range(BRAC_BEHAV_STEPS):
                    # Sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    loss = agent.update_behavior_policy_clone(batch=batch)
                    if i % 10 == 0:
                        i_fmt = str(i).zfill(math.floor(math.log10(BRAC_BEHAV_STEPS)) + 1)
                        logger.info(f"bc iter: {i_fmt} | bc loss: {loss}")

        with timed('training'):
            for training_step in range(args.training_steps_per_iter):

                if agent.param_noise is not None:
                    if training_step % args.pn_adapt_frequency == 0:
                        # Adapt parameter noise
                        agent.adapt_param_noise()
                    if rank == 0 and iters_so_far % args.eval_frequency == 0:
                        # Store the action-space dist between perturbed and non-perturbed
                        d['pn_dist'].append(agent.pn_dist)
                        # Store the new std resulting from the adaption
                        d['pn_cur_std'].append(agent.param_noise.cur_std)

                # Sample a batch of transitions from the replay buffer
                batch = agent.sample_batch()

                # Update the actor and critic
                metrics, lrnows = agent.update_actor_critic(
                    batch=batch,
                    update_actor=not bool(iters_so_far % args.actor_update_delay),  # from TD3
                    iters_so_far=iters_so_far,
                )
                if rank == 0 and iters_so_far % args.eval_frequency == 0:
                    # Log training stats
                    d['actr_losses'].append(metrics['actr_loss'])
                    d['crit_losses'].append(metrics['crit_loss'])
                    if iters_so_far >= agent.hps.warm_start:
                        d['q_stats_mean'].append(metrics['q_mean'])
                        d['q_stats_std'].append(metrics['q_std'])
                        d['q_stats_min'].append(metrics['q_min'])
                        d['q_stats_max'].append(metrics['q_max'])
                        if args.algo.split('_')[0] == 'ptso':
                            if agent.hps.ptso_use_or_monitor_grad_pen:
                                d['phi_gradnorm_s'].append(metrics['phi_gradnorm_s'])
                                d['phi_gradnorm_a'].append(metrics['phi_gradnorm_a'])
                            d['u_stats_mean'].append(metrics['u_mean'])
                            d['u_stats_std'].append(metrics['u_std'])
                            d['u_stats_min'].append(metrics['u_min'])
                            d['u_stats_max'].append(metrics['u_max'])
                            d['v_q'].append(metrics['v_q'])
                            d['u_pred'].append(metrics['u_pred'])
                            d['u_losses'].append(metrics['u_loss'])  # careful, key mismatch
                            if agent.hps.ptso_use_rnd_monitoring:
                                d['rnd_score'].append(metrics['rnd_score'])
                            d['v_q_uniform'].append(metrics['v_q_uniform'])
                            d['u_pred_uniform'].append(metrics['u_pred_uniform'])
                    if agent.hps.clipped_double:
                        d['twin_losses'].append(metrics['twin_loss'])
                    if agent.hps.prioritized_replay:
                        iws = metrics['iws']  # last one only

        if rank == 0 and iters_so_far % args.eval_frequency == 0:

            with timed("evaluating"):

                # Update the parameters of the eval actors with the ones of the trained one
                agent.update_eval_nets()

                for eval_step in range(args.eval_steps_per_iter):
                    # Sample an episode
                    main_eval_ep = main_eval_ep_gen.__next__()
                    maxq_eval_ep = maxq_eval_ep_gen.__next__()
                    # Aggregate data collected during the evaluation to the buffers
                    d['main_eval_len'].append(main_eval_ep['ep_len'])
                    d['maxq_eval_len'].append(maxq_eval_ep['ep_len'])
                    d['main_eval_env_ret'].append(main_eval_ep['ep_env_ret'])
                    d['maxq_eval_env_ret'].append(maxq_eval_ep['ep_env_ret'])
                main_eval_deque.append(np.mean(d['main_eval_env_ret']))
                maxq_eval_deque.append(np.mean(d['maxq_eval_env_ret']))

        # Increment counters
        iters_so_far += 1
        if args.offline:
            _str = 'iters so far'
            step = copy(iters_so_far)
        else:
            timesteps_so_far += args.rollout_len
            _str = 'timesteps so far'
            step = copy(timesteps_so_far)

        if rank == 0 and ((iters_so_far - 1) % args.eval_frequency == 0):

            # Log stats in csv
            logger.record_tabular(_str, step)

            if iters_so_far - 1 >= agent.hps.warm_start:
                logger.record_tabular('q_stats_mean', np.mean(d['q_stats_mean']))
                logger.record_tabular('q_stats_std', np.mean(d['q_stats_std']))
                logger.record_tabular('q_stats_min', np.mean(d['q_stats_min']))
                logger.record_tabular('q_stats_max', np.mean(d['q_stats_max']))
                if args.algo.split('_')[0] == 'ptso':
                    if agent.hps.ptso_use_or_monitor_grad_pen:
                        logger.record_tabular('phi_gradnorm_s', np.mean(d['phi_gradnorm_s']))
                        logger.record_tabular('phi_gradnorm_a', np.mean(d['phi_gradnorm_a']))
                    logger.record_tabular('u_stats_mean', np.mean(d['u_stats_mean']))
                    logger.record_tabular('u_stats_std', np.mean(d['u_stats_std']))
                    logger.record_tabular('u_stats_min', np.mean(d['u_stats_min']))
                    logger.record_tabular('u_stats_max', np.mean(d['u_stats_max']))
                    logger.record_tabular('v_q', np.mean(d['v_q']))
                    logger.record_tabular('u_pred', np.mean(d['u_pred']))
                    if agent.hps.ptso_use_rnd_monitoring:
                        logger.record_tabular('rnd_score', np.mean(d['rnd_score']))
                    logger.record_tabular('v_q_uniform', np.mean(d['v_q_uniform']))
                    logger.record_tabular('u_pred_uniform', np.mean(d['u_pred_uniform']))
            logger.record_tabular('main_eval_len', np.mean(d['main_eval_len']))
            logger.record_tabular('maxq_eval_len', np.mean(d['maxq_eval_len']))
            logger.record_tabular('main_eval_env_ret', np.mean(d['main_eval_env_ret']))
            logger.record_tabular('maxq_eval_env_ret', np.mean(d['maxq_eval_env_ret']))
            logger.record_tabular('avg_main_eval_env_ret', np.mean(main_eval_deque))
            logger.record_tabular('avg_maxq_eval_env_ret', np.mean(maxq_eval_deque))
            logger.info("dumping stats in .csv file")
            logger.dump_tabular()

            if args.record:
                # Record the last episode in a video
                record_video(vid_dir, iters_so_far, maxq_eval_ep['obs_render'])  # picked only one, maxq

            # Log stats in dashboard
            if agent.hps.prioritized_replay:
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                np.quantile(iws, quantiles)
                wandb.log({"q{}".format(q): np.quantile(iws, q)
                           for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                          step=step)
            if agent.param_noise is not None:
                wandb.log({'pn_dist': np.mean(d['pn_dist']),
                           'pn_cur_std': np.mean(d['pn_cur_std'])},
                          step=step)
            wandb.log({'actr_loss': np.mean(d['actr_losses']),
                       'actr_lrnow': np.array(lrnows['actr']),
                       'crit_loss': np.mean(d['crit_losses'])},
                      step=step)
            if agent.hps.clipped_double:
                wandb.log({'twin_loss': np.mean(d['twin_losses'])},
                          step=step)
            if iters_so_far - 1 >= agent.hps.warm_start:
                wandb.log({'q_stats_mean': np.mean(d['q_stats_mean']),
                           'q_stats_std': np.mean(d['q_stats_std']),
                           'q_stats_min': np.mean(d['q_stats_min']),
                           'q_stats_max': np.mean(d['q_stats_max'])},
                          step=step)
                if args.algo.split('_')[0] == 'ptso':
                    wandb.log({'u_stats_mean': np.mean(d['u_stats_mean']),
                               'u_stats_std': np.mean(d['u_stats_std']),
                               'u_stats_min': np.mean(d['u_stats_min']),
                               'u_stats_max': np.mean(d['u_stats_max']),
                               'v_q': np.mean(d['v_q']),
                               'u_pred': np.mean(d['u_pred']),
                               'u_loss': np.mean(d['u_losses']),
                               'v_q_uniform': np.mean(d['v_q_uniform']),
                               'u_pred_uniform': np.mean(d['u_pred_uniform'])},
                              step=step)
                    if agent.hps.ptso_use_or_monitor_grad_pen:
                        wandb.log({'phi_gradnorm_s': np.mean(d['phi_gradnorm_s']),
                                   'phi_gradnorm_a': np.mean(d['phi_gradnorm_a'])},
                                  step=step)
                    if agent.hps.ptso_use_rnd_monitoring:
                        wandb.log({'rnd_score': np.mean(d['rnd_score'])},
                                  step=step)
            wandb.log({'main_eval_len': np.mean(d['main_eval_len']),
                       'maxq_eval_len': np.mean(d['maxq_eval_len']),
                       'main_eval_env_ret': np.mean(d['main_eval_env_ret']),
                       'maxq_eval_env_ret': np.mean(d['maxq_eval_env_ret']),
                       'avg_main_eval_env_ret': np.mean(main_eval_deque),
                       'avg_maxq_eval_env_ret': np.mean(maxq_eval_deque)},
                      step=step)

            # Clear the iteration's running stats
            d.clear()

    if not agent.hps.offline:
        # When running the algorithm online, save behavior policy model once we are done iterating
        if rank == 0:
            # Save the model
            agent.save(ckpt_dir, iters_so_far)
            logger.info("we're done. Saving behavior policy model @: {}".format(ckpt_dir))
            logger.info("bye.")
            # Save the replay buffer and the full history
            agent.save_memory(replay_dir, iters_so_far)
            agent.save_full_history(replay_dir, iters_so_far)

    if rank == 0:
        # Save once we are done iterating
        agent.save(ckpt_dir, iters_so_far)
        logger.info("we're done. Saving model @: {}".format(ckpt_dir))
        logger.info("bye.")
