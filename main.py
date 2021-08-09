import os
import random
from copy import copy
from collections import defaultdict

from mpi4py import MPI
import numpy as np
import torch

import logger
from helpers.argparsers import argparser
from helpers.experiment import ExperimentInitializer
from helpers.math_util import discount
from helpers.distributed_util import setup_mpi_gpus
from helpers.env_makers import make_env
from agents import orchestrator
from agents.ddpg_agent import DDPGAgent
from agents.sac_agent import SACAgent
from agents.bcq_agent import BCQAgent
from agents.bear_agent import BEARAgent
from agents.brac_agent import BRACAgent
from agents.cql_agent import CQLAgent
from agents.bcp_agent import BCPAgent
from helpers import h5_util as H


SKIP_TIMEOUT_TRANSITIONS = True


def extract_dataset(args, env, dataset_path):
    RMIN = np.infty
    RMAX = -np.infty
    # Ensure the environment possesses a dataset in the D4RL suite
    assert hasattr(env, 'get_dataset')  # unique (a priori) to d4rl envs
    # Load the offline dataset
    _dataset = H.load_dict_h5py(dataset_path)
    to_load_in_memory = copy(_dataset)
    # Rename the keys to the ones the rest of the codebase use
    to_load_in_memory['obs0'] = to_load_in_memory.pop('observations')
    to_load_in_memory['acs'] = to_load_in_memory.pop('actions')
    to_load_in_memory['rews'] = to_load_in_memory.pop('rewards')
    to_load_in_memory['dones1'] = to_load_in_memory.pop('terminals')
    # Augment with the 'obs1' key, adding the next observation in every transition
    _obs0 = []
    _acs = []
    _obs1 = []
    _acs1 = []
    _rews = []
    _dones1 = []
    ep_step = 0
    for i in range(to_load_in_memory['rews'].shape[0] - 1):  # key arbitrarily chosen
        ob = to_load_in_memory['obs0'][i]
        ac = to_load_in_memory['acs'][i]
        next_ob = to_load_in_memory['obs0'][i+1]
        next_ac = to_load_in_memory['acs'][i+1]
        rew = to_load_in_memory['rews'][i]
        done = bool(to_load_in_memory['dones1'][i])
        # Treat termination cases appropriately
        final_timestep = (ep_step == env._max_episode_steps - 1)
        if SKIP_TIMEOUT_TRANSITIONS and final_timestep:
            # Exclude transitions that terminate by timeout
            ep_step = 0
            continue
        if done or final_timestep:
            ep_step = 0
        # Add the transition to the dataset
        _obs0.append(ob)
        _acs.append(ac)
        _obs1.append(next_ob)
        _acs1.append(next_ac)
        RMIN = min(RMIN, rew)
        RMAX = max(RMAX, rew)
        _rews.append(rew)
        _dones1.append(done)
        ep_step += 1
    # Overwrite the content of the dataset
    to_load_in_memory['obs0'] = _obs0
    to_load_in_memory['acs'] = _acs
    to_load_in_memory['obs1'] = _obs1
    to_load_in_memory['acs1'] = _acs1
    to_load_in_memory['rews'] = _rews
    to_load_in_memory['dones1'] = _dones1
    # Wrap each value into a numpy array
    to_load_in_memory = {k: np.array(v) for k, v in to_load_in_memory.items()}
    ini_num_transitions = to_load_in_memory['rews'].shape[0]
    logger.info(f"the dataset contains {ini_num_transitions} transitions")

    # We now create a list of all the complete trajectories present in the dataset,
    # calculate the Monte-Carlo return for each transition, and insert them with their own key.
    # Initialize the list of complete trajectories in the dataset, and first trajectory to fill
    trajectory_list = []
    trajectory = defaultdict(list)
    episode_step = 0
    for i in range(to_load_in_memory['rews'].shape[0]):  # key arbitrarily chosen
        # Define termination triggers, due to timeout or terminating the episode through the MDP
        done_bool = bool(to_load_in_memory['dones1'][i])
        final_timestep = (episode_step == env._max_episode_steps - 1)
        # When arrived at the end of the episode, add the completed trajectory to the list
        if done_bool or final_timestep:
            episode_step = 0
            np_trajectory = {}
            for k in to_load_in_memory:
                np_trajectory[k] = np.array(trajectory[k])
            # Add a key for the MC return and populate it with it
            np_trajectory['rets'] = discount(np_trajectory['rews'], args.gamma)
            # Add the formated trajectory to the list of trajectories
            trajectory_list.append(np_trajectory)
            # Initialize the next trajectory
            trajectory = defaultdict(list)
        # Add the transition to the current trajectory's dictionaries
        for k in to_load_in_memory:
            trajectory[k].append(to_load_in_memory[k][i])
        episode_step += 1
    logger.info(f"the dataset contains {len(trajectory_list)} completed trajectories")
    logger.info(f"the dataset contains {len(trajectory['rews'])} orphan transitions")  # key arbitrarily chosen

    # Now that we have the Monte-Carlo returns associated with each transition of complete trajectories,
    # we create a new dictionary containing only the data from these conplete trajectories, and leave
    # the previous disctionaries intact, just in case we need them at some point.
    # Note, if the last trajectory in the dataset is not complete (does not terminate), then it is discarded.
    to_load_in_memory_only_completed = defaultdict(list)
    for i in range(len(trajectory_list)):  # needs to be like this apparently, exception thing
        for k in list(to_load_in_memory.keys()) + ['rets']:
            to_load_in_memory_only_completed[k].extend(trajectory_list[i][k])
    to_load_in_memory_only_completed = {k: np.array(v) for k, v in to_load_in_memory_only_completed.items()}
    # Truncate the lists of the original dataset to only contain data from completed trajectories,
    # which therefore only contain transitions augmented with Monte-Carlo returns.
    for k in to_load_in_memory:
        to_load_in_memory[k] = to_load_in_memory[k][:to_load_in_memory_only_completed[k].shape[0]]
        # Verify that the slicing wroked as intended
        assert np.all(to_load_in_memory_only_completed[k] == to_load_in_memory[k]), "size issue."
    new_num_transitions = to_load_in_memory['rews'].shape[0]
    logger.info(f"the dataset contains {new_num_transitions} transitions after removing orphans")
    # Now that they are the same size, directly transfer the the  'rets' key and value to the original dataset
    to_load_in_memory['rets'] = to_load_in_memory_only_completed['rets']

    # Carry out anity-check on the shapes of what's in the dataset
    canon_size = to_load_in_memory['rews'].shape[0]  # key arbitrarily chosen
    for k, v in to_load_in_memory.items():
        logger.info(f"key({k}) -> shape({v.shape})")
        assert v.shape[0] == canon_size

    # Summarize the changes
    logger.info(f"the dataset went through these size changes: {ini_num_transitions} -> {new_num_transitions}")
    # Define new memory size
    logger.info(f"over-writting memory size: {copy(args.mem_size)} -> {new_num_transitions}")
    # Log the range of the reward signal in the dataset
    logger.info(f"RMIN={RMIN}, RMAX={RMAX}")

    return to_load_in_memory, new_num_transitions


def train(args):
    """Train an agent"""

    # Get the current process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args.algo = args.algo + '_' + str(world_size).zfill(3)

    torch.set_num_threads(1)

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, rank=rank, world_size=world_size)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_name()

    # Set device-related knobs
    if args.cuda:
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(f"cuda:{args.gpu_index}")
        setup_mpi_gpus()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
        device = torch.device("cpu")
    args.device = device  # add the device to hps for convenience
    logger.info("device in use: {}".format(device))

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    worker_seed = args.seed + (1000000 * (rank + 1))
    eval_seed = args.seed + 1000000

    # Create environment
    env = make_env(args.env_id, worker_seed)

    if args.offline:

        # Extract the dataset from the archive
        to_load_in_memory, new_num_transitions = extract_dataset(args, env, args.dataset_path)
        # Overwrite the memory size hyper-parameter in the offline setting
        args.mem_size = new_num_transitions

        if args.mix_with_random:
            # Extract the dataset containing random data from the same environment
            assert 0. <= args.mixing_ratio <= 1., "ratio must be between 0 and 1 (bounds included)."
            dpath = args.dataset_path
            filename = dpath.split('/')[-1]
            filename_no_ext, file_ext = filename.split('.')
            splits = filename_no_ext.split('-')
            version_suffix = splits[-1]
            env_name = splits[0]
            assert env_name in ['halfcheetah', 'hopper', 'walker2d']
            env_type = splits[1:-1]
            assert 'expert' in env_type, "environment name must contain 'expert'."
            new_env = '-'.join([env_name, 'random', version_suffix])
            _new_path = '/' + os.path.join(*dpath.split('/')[:-1], new_env + '.' + file_ext)
            _new_env = make_env(new_env, worker_seed)
            new_to_load_in_memory, _ = extract_dataset(args, _new_env, _new_path)

            # Sanity-check: verify that the keys coincide
            assert list(to_load_in_memory.keys()) == list(new_to_load_in_memory.keys()), "keys mismatch."
            # Shuffle both datasets, which uses the previously-set random seed
            for k in to_load_in_memory:
                # Numpy's shuffle operations are in-place, and by default only along the first axis
                random.Random(args.seed).shuffle(to_load_in_memory[k])
                random.Random(args.seed).shuffle(new_to_load_in_memory[k])
            # Note, we verified that the two datasets have the same keys right above

            # Merge the datasets
            highest_index = to_load_in_memory['rews'].shape[0] * args.mixing_ratio  # key arbitrarily chosen
            highest_index = min(int(np.floor(highest_index)), new_to_load_in_memory['rews'].shape[0])  # same
            for k in to_load_in_memory.keys():
                old_size = to_load_in_memory[k].shape[0]
                to_load_in_memory[k] = np.concatenate([to_load_in_memory[k][highest_index:, ...],
                                                       new_to_load_in_memory[k][0:highest_index, ...]], axis=0)
                new_size = to_load_in_memory[k].shape[0]
                assert new_size == old_size, "size mismatch."
                logger.info(f"post-merge | key({k}) -> shape({to_load_in_memory[k].shape})")

            # Schuffle the assembled dataset
            for k in to_load_in_memory.keys():
                # Numpy's shuffle operations are in-place, and by default only along the first axis
                random.Random(args.seed).shuffle(to_load_in_memory[k])
    else:
        to_load_in_memory = None

    # Create an agent wrapper
    if args.algo.split('_')[0] == 'ddpg':
        def agent_wrapper():
            return DDPGAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

    elif args.algo.split('_')[0] == 'sac':
        def agent_wrapper():
            return SACAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

    elif args.algo.split('_')[0] == 'bcq':
        def agent_wrapper():
            return BCQAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

        assert args.offline

    elif args.algo.split('_')[0] == 'bear':
        def agent_wrapper():
            return BEARAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

        assert args.offline

    elif args.algo.split('_')[0] == 'brac':
        def agent_wrapper():
            return BRACAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

        assert args.offline

    elif args.algo.split('_')[0] == 'cql':
        def agent_wrapper():
            return CQLAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

        assert args.offline

    elif args.algo.split('_')[0] == 'bcp':
        def agent_wrapper():
            return BCPAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

        assert args.offline

    else:
        raise NotImplementedError("algorithm not covered")

    # Create an evaluation environment not to mess up with training rollouts
    main_eval_env = None
    maxq_eval_env = None
    cwpq_eval_env = None
    if rank == 0:
        main_eval_env = make_env(args.env_id, eval_seed)
        maxq_eval_env = make_env(args.env_id, eval_seed)
        cwpq_eval_env = make_env(args.env_id, eval_seed)

    # Train
    orchestrator.learn(
        args=args,
        rank=rank,
        env=env,
        main_eval_env=main_eval_env,
        maxq_eval_env=maxq_eval_env,
        cwpq_eval_env=cwpq_eval_env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
        use_noise_process=(args.algo == 'ddpg' and not args.offline),
    )

    # Close environment
    env.close()

    # Close the eval env
    if rank == 0:
        main_eval_env.close()
        maxq_eval_env.close()


def evaluate(args):
    """Evaluate an agent"""

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_name()

    # Seedify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = make_env(args.env_id, args.seed)

    # Create an agent wrapper
    if args.algo == 'ddpg':
        def agent_wrapper():
            return DDPGAgent(
                env=env,
                device='cpu',
                hps=args,
                to_load_in_memory=None,
            )

    elif args.algo == 'sac':
        def agent_wrapper():
            return SACAgent(
                env=env,
                device='cpu',
                hps=args,
                to_load_in_memory=None,
            )

    else:
        raise NotImplementedError("algorithm not covered")

    # Evaluate
    orchestrator.evaluate(
        args=args,
        env=env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
    )

    # Close environment
    env.close()


def generate(args):
    """Generate replay buffer from an agent trained online"""

    assert (args.algo == 'ddpg' or args.algo == 'sac') and not args.offline

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()
    # Create experiment name
    experiment_name = experiment.get_name()

    # Seedify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = make_env(args.env_id, args.seed)

    # Create an agent wrapper
    if args.algo == 'ddpg':
        def agent_wrapper():
            return DDPGAgent(
                env=env,
                device='cpu',
                hps=args,
                to_load_in_memory=None,
            )

    elif args.algo == 'sac':
        def agent_wrapper():
            return SACAgent(
                env=env,
                device='cpu',
                hps=args,
                to_load_in_memory=None,
            )

    else:
        raise NotImplementedError("algorithm not covered")

    # Evaluate
    orchestrator.generate_buffer(
        args=args,
        env=env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
    )

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = argparser().parse_args()

    # Make the paths absolute
    _args.root = os.path.dirname(os.path.abspath(__file__))
    for k in ['checkpoints', 'logs', 'videos', 'replays']:
        new_k = "{}_dir".format(k[:-1])
        vars(_args)[new_k] = os.path.join(_args.root, 'data', k)

    if _args.task == 'train':
        train(_args)
    elif _args.task == 'eval':
        evaluate(_args)
    elif _args.task == 'generate':
        generate(_args)
    else:
        raise NotImplementedError
