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
from helpers import h5_util as U


def train(args):
    """Train an agent"""

    # FIXME
    args.algo = 'ddpg'
    args.behavior_dir = None
    args.offline = False
    args.use_expert_demos = False
    args.buffer_path = None
    args.sub_rate = 20
    # FIXME

    # Make the paths absolute
    args.root = os.path.dirname(os.path.abspath(__file__))
    for k in ['checkpoints', 'logs', 'videos', 'behavior']:
        new_k = "{}_dir".format(k[:-1])
        vars(args)[new_k] = os.path.join(args.root, 'data', k)

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

    if args.offine:
        if args.use_expert_demos:
            to_load_in_memory = DemoDataset(
                expert_path=args.expert_path,
                num_demos=args.num_demos,
                env=env,
                wrap_absorb=args.wrap_absorb,
                sub_rate=args.sub_rate,
            ).data
        else:
            to_load_in_memory = U.load_dict_h5py(args.buffer_path)
        # Make sure the loaded data structure is a disctionary
        assert isinstance(to_load_in_memory, dict), "must be a dictionary"
        # Overwrite the memory size hyper-parameter in the offline setting
        old_memory_size = copy(args.memory_size)
        args.memory_size = to_load_in_memory['rews'].shape[0]  # key arbitrarily chosen
        logger.info("over-written memory size: {} -> {}".format(old_memory_size, args.memory_size))
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

    else:
        raise NotImplementedError("algorithm not covered")

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(args.env_id, eval_seed)

    # Train
    orchestrator.learn(args=args,
                       rank=rank,
                       device=device,
                       env=env,
                       eval_env=eval_env,
                       experiment_name=experiment_name,
                       to_load_in_memory=to_load_in_memory)

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

    # Seedify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create environment
    env = make_env(args.env_id, args.seed)

    # Evaluate agent trained via DDPG
    orchestrator.evaluate(args=args,
                          device='cpu',
                          env=env)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = argparser().parse_args()
    if _args.task == 'train':
        train(_args)
    elif _args.task == 'eval':
        evaluate(_args)
    else:
        raise NotImplementedError
