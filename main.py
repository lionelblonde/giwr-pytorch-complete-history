import os
import random
from copy import copy

from mpi4py import MPI
import numpy as np
import torch

from helpers import logger
from helpers.argparsers import argparser
from helpers.experiment import ExperimentInitializer
from helpers.distributed_util import setup_mpi_gpus
from helpers.env_makers import make_env
from agents import orchestrator
from helpers.dataset import DemoDataset
from agents.ddpg_agent import DDPGAgent
from agents.sac_agent import SACAgent
from agents.bcq_agent import BCQAgent
from agents.bear_agent import BEARAgent
from helpers import h5_util as H


def train(args):
    """Train an agent"""

    # Get the current process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

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
        device = torch.device("cuda:0")
        setup_mpi_gpus()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
        device = torch.device("cpu")
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
        # Load the offline dataset
        if hasattr(env, 'get_dataset'):  # unique (a priori) to d4rl envs
            to_load_in_memory = H.load_dict_h5py(args.dataset_path)
            # Rename the keys to the ones the rest of the codebase use
            to_load_in_memory['obs0'] = to_load_in_memory.pop('observations')
            to_load_in_memory['acs'] = to_load_in_memory.pop('actions')
            to_load_in_memory['rews'] = to_load_in_memory.pop('rewards')
            to_load_in_memory['dones1'] = to_load_in_memory.pop('terminals')
            # Augment with the 'obs1' key, adding the next observation in every transition
            _obs0 = []
            _acs = []
            _obs1 = []
            _rews = []
            _dones1 = []
            ep_step = 0
            for i in range(to_load_in_memory['rews'].shape[0] - 1):  # key arbitrarily chosen
                ob = to_load_in_memory['obs0'][i]
                ac = to_load_in_memory['acs'][i]
                next_ob = to_load_in_memory['obs0'][i+1]
                rew = to_load_in_memory['rews'][i]
                done = bool(to_load_in_memory['dones1'][i])
                # Treat termination cases appropriately
                final_timestep = (ep_step == env._max_episode_steps - 1)
                if final_timestep or done:
                    ep_step = 0
                    if final_timestep:
                        # Exclude transitions that terminate by timeout
                        continue
                # Add the transition to the dataset
                _obs0.append(ob)
                _acs.append(ac)
                _obs1.append(next_ob)
                _rews.append(rew)
                _dones1.append(done)
                ep_step += 1
            # Overwrite the content of the dataset
            to_load_in_memory['obs0'] = _obs0
            to_load_in_memory['acs'] = _acs
            to_load_in_memory['obs1'] = _obs1
            to_load_in_memory['rews'] = _rews
            to_load_in_memory['dones1'] = _dones1
            # Wrap each value into a numpy array
            to_load_in_memory = {k: np.array(v) for k, v in to_load_in_memory.items()}
        else:
            if args.use_expert_demos:
                to_load_in_memory = DemoDataset(
                    expert_path=args.expert_path,
                    num_demos=args.num_demos,
                    env=env,
                    wrap_absorb=args.wrap_absorb,
                    sub_rate=args.sub_rate,
                ).data
            else:
                to_load_in_memory = H.load_dict_h5py(args.dataset_path)
        # Make sure the loaded data structure is a disctionary
        assert isinstance(to_load_in_memory, dict), "must be a dictionary"
        # Overwrite the memory size hyper-parameter in the offline setting
        old_memory_size = copy(args.mem_size)
        args.mem_size = to_load_in_memory['rews'].shape[0]  # key arbitrarily chosen
        logger.info("over-written memory size: {} -> {}".format(old_memory_size, args.mem_size))
    else:
        to_load_in_memory = None

    # Create an agent wrapper
    if args.algo == 'ddpg':
        def agent_wrapper():
            return DDPGAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

    elif args.algo == 'sac':
        def agent_wrapper():
            return SACAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

    elif args.algo == 'bcq':
        def agent_wrapper():
            return BCQAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

        assert args.offline

    elif args.algo == 'bear':
        def agent_wrapper():
            return BEARAgent(
                env=env,
                device=device,
                hps=args,
                to_load_in_memory=to_load_in_memory,
            )

        assert args.offline

    else:
        raise NotImplementedError("algorithm not covered")

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(args.env_id, eval_seed)

    # Train
    orchestrator.learn(
        args=args,
        rank=rank,
        env=env,
        eval_env=eval_env,
        agent_wrapper=agent_wrapper,
        experiment_name=experiment_name,
        use_noise_process=(args.algo == 'ddpg' and not args.offline),
    )

    # Close environment
    env.close()

    # Close the eval env
    if eval_env is not None:
        assert rank == 0
        eval_env.close()


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
    for k in ['checkpoints', 'logs', 'videos', 'replays', 'explorations']:
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
