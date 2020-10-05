import argparse
from copy import deepcopy
import os
import sys
import numpy as np
import subprocess
import yaml

from helpers import logger
from helpers.misc_util import zipsame, boolean_flag
from helpers.experiment import uuid as create_uuid


parser = argparse.ArgumentParser(description="Job Spawner")
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--envset', type=str, default=None)
parser.add_argument('--num_demos', '--list', nargs='+', type=str, default=None)
parser.add_argument('--caliber', type=str, default=None)
boolean_flag(parser, 'call', default=False, help="launch immediately?")
boolean_flag(parser, 'sweep', default=False, help="hp search?")
boolean_flag(parser, 'wandb_upgrade', default=True, help="upgrade wandb?")
args = parser.parse_args()

# Retrieve config from filesystem
CONFIG = yaml.safe_load(open(args.config))

# Extract parameters from config
NUM_SEEDS = CONFIG['parameters']['num_seeds']
NEED_DEMOS = CONFIG['parameters']['use_expert_demos']
assert not NEED_DEMOS or CONFIG['parameters']['offline']
NEED_DSETS = False
if NEED_DEMOS:
    NUM_DEMOS = [int(i) for i in args.num_demos]
else:
    NUM_DEMOS = [0]  # arbitrary, only used for dim checking
    # If we are not using demos but are in the offline setting, need datasets
    if CONFIG['parameters']['offline']:
        NEED_DSETS = True
CLUSTER = CONFIG['resources']['cluster']
WANDB_PROJECT = CONFIG['resources']['wandb_project'].upper() + '-' + CLUSTER.upper()
CONDA = CONFIG['resources']['conda_env']
# Define experiment type
TYPE = 'sweep' if args.sweep else 'fixed'
# Write out the boolean arguments (using the 'boolean_flag' function)
BOOL_ARGS = ['cuda', 'render', 'record', 'with_scheduler',
             'layer_norm',
             'prioritized_replay', 'ranked', 'unreal',
             'n_step_returns', 'ret_norm', 'popart',
             'clipped_double', 'targ_actor_smoothing',
             'use_c51', 'use_qr',
             'offline', 'use_expert_demos',
             'state_dependent_std', 'use_adaptive_alpha']

# Create the list of environments from the indicated benchmark
BENCH = CONFIG['parameters']['benchmark']

# Define the caliber got the distribution scheme
if args.caliber == 'monoshort':
    PARTITION = 'shared-EL7'
    NUM_WORKERS = 1
    TIMEOUT = '0-12:00:00'
elif args.caliber == 'monolong':
    PARTITION = 'mono-EL7'
    NUM_WORKERS = 1
    TIMEOUT = '2-00:00:00'
elif args.caliber == 'monoverylong':
    PARTITION = 'mono-EL7'
    NUM_WORKERS = 1
    TIMEOUT = '4-00:00:00'
elif args.caliber == 'multishort':
    PARTITION = 'shared-EL7'
    NUM_WORKERS = 16
    TIMEOUT = '0-12:00:00'
elif args.caliber == 'multilong':
    PARTITION = 'mono-EL7'
    NUM_WORKERS = 16
    TIMEOUT = '2-00:00:00'
elif args.caliber == 'multiverylong':
    PARTITION = 'mono-EL7'
    NUM_WORKERS = 16
    TIMEOUT = '4-00:00:00'
else:
    raise ValueError("invalid caliber")


if BENCH == 'mujoco':

    # Define environments map
    TOC = {
        'debug': ['Hopper-v3'],
        'flareon': ['InvertedPendulum-v2',
                    'InvertedDoublePendulum-v2',
                    'Hopper-v3'],
        'glaceon': ['Hopper-v3',
                    'Walker2d-v3',
                    'HalfCheetah-v3',
                    'Ant-v3'],
        'humanoid': ['Humanoid-v3'],
        'ant': ['Ant-v3'],
        'suite': ['InvertedDoublePendulum-v2',
                  'Hopper-v3',
                  'Walker2d-v3',
                  'HalfCheetah-v3',
                  'Ant-v3'],
    }
    ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environement partitions map
        PEP = {
            'InvertedPendulum': 'shared-EL7',
            'Reacher': 'shared-EL7',
            'InvertedDoublePendulum': 'shared-EL7',
            'Hopper': PARTITION,
            'Walker2d': PARTITION,
            'HalfCheetah': PARTITION,
            'Ant': PARTITION,
            'Humanoid': PARTITION,
        }
        # Define per-environment ntasks map
        PEC = {
            'InvertedPendulum': 8,
            'Reacher': 8,
            'InvertedDoublePendulum': 8,
            'Hopper': NUM_WORKERS,
            'Walker2d': NUM_WORKERS,
            'HalfCheetah': NUM_WORKERS,
            'Ant': NUM_WORKERS,
            'Humanoid': NUM_WORKERS,
        }
        # Define per-environment timeouts map
        PET = {
            'InvertedPendulum': '0-06:00:00',
            'Reacher': '0-06:00:00',
            'InvertedDoublePendulum': '0-06:00:00',
            'Hopper': TIMEOUT,
            'Walker2d': TIMEOUT,
            'HalfCheetah': TIMEOUT,
            'Ant': TIMEOUT,
            'Humanoid': TIMEOUT,
        }

elif BENCH == 'dmc':

    TOC = {
        'debug': ['Hopper-Hop-Feat-v0'],
        'flareon': ['Hopper-Hop-Feat-v0',
                    'Walker-Run-Feat-v0'],
        'glaceon': ['Hopper-Hop-Feat-v0',
                    'Cheetah-Run-Feat-v0',
                    'Walker-Run-Feat-v0'],
        'stacker': ['Stacker-Stack_2-Feat-v0',
                    'Stacker-Stack_4-Feat-v0'],
        'humanoid': ['Humanoid-Walk-Feat-v0',
                     'Humanoid-Run-Feat-v0'],
        'cmu': ['Humanoid_CMU-Stand-Feat-v0',
                'Humanoid_CMU-Run-Feat-v0'],
        'quad': ['Quadruped-Walk-Feat-v0',
                 'Quadruped-Run-Feat-v0',
                 'Quadruped-Escape-Feat-v0',
                 'Quadruped-Fetch-Feat-v0'],
        'dog': ['Dog-Run-Feat-v0',
                'Dog-Fetch-Feat-v0'],
        'suite': ['Hopper-Hop-Feat-v0',
                  'Cheetah-Run-Feat-v0',
                  'Walker-Run-Feat-v0',
                  'Quadruped-Walk-Feat-v0',
                  'Quadruped-Run-Feat-v0',
                  'Quadruped-Escape-Feat-v0',
                  'Quadruped-Fetch-Feat-v0'],
    }
    ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environement partitions map
        PEP = {
            'Hopper-Hop-Feat': PARTITION,
            'Walker-Run-Feat': PARTITION,
            'Cheetah-Run-Feat': PARTITION,
            'Stacker-Stack_2-Feat': PARTITION,
            'Stacker-Stack_4-Feat': PARTITION,
            'Humanoid-Walk-Feat': PARTITION,
            'Humanoid-Run-Feat': PARTITION,
            'Humanoid_CMU-Stand-Feat': PARTITION,
            'Humanoid_CMU-Run-Feat': PARTITION,
            'Quadruped-Walk-Feat': PARTITION,
            'Quadruped-Run-Feat': PARTITION,
            'Quadruped-Escape-Feat': PARTITION,
            'Quadruped-Fetch-Feat': PARTITION,
            'Dog-Run-Feat': PARTITION,
            'Dog-Fetch-Feat': PARTITION,
        }
        # Define per-environment ntasks map
        PEC = {
            'Hopper-Hop-Feat': NUM_WORKERS,
            'Walker-Run-Feat': NUM_WORKERS,
            'Cheetah-Run-Feat': NUM_WORKERS,
            'Stacker-Stack_2-Feat': NUM_WORKERS,
            'Stacker-Stack_4-Feat': NUM_WORKERS,
            'Humanoid-Walk-Feat': NUM_WORKERS,
            'Humanoid-Run-Feat': NUM_WORKERS,
            'Humanoid_CMU-Stand-Feat': NUM_WORKERS,
            'Humanoid_CMU-Run-Feat': NUM_WORKERS,
            'Quadruped-Walk-Feat': NUM_WORKERS,
            'Quadruped-Run-Feat': NUM_WORKERS,
            'Quadruped-Escape-Feat': NUM_WORKERS,
            'Quadruped-Fetch-Feat': NUM_WORKERS,
            'Dog-Run-Feat': NUM_WORKERS,
            'Dog-Fetch-Feat': NUM_WORKERS,
        }
        # Define per-environment timeouts map
        PET = {
            'Hopper-Hop-Feat': TIMEOUT,
            'Walker-Run-Feat': TIMEOUT,
            'Cheetah-Run-Feat': TIMEOUT,
            'Stacker-Stack_2-Feat': TIMEOUT,
            'Stacker-Stack_4-Feat': TIMEOUT,
            'Humanoid-Walk-Feat': TIMEOUT,
            'Humanoid-Run-Feat': TIMEOUT,
            'Humanoid_CMU-Stand-Feat': TIMEOUT,
            'Humanoid_CMU-Run-Feat': TIMEOUT,
            'Quadruped-Walk-Feat': TIMEOUT,
            'Quadruped-Run-Feat': TIMEOUT,
            'Quadruped-Escape-Feat': TIMEOUT,
            'Quadruped-Fetch-Feat': TIMEOUT,
            'Dog-Run-Feat': TIMEOUT,
            'Dog-Fetch-Feat': TIMEOUT,
        }

elif BENCH == 'd4rl':

    TOC = {
        'debug': ['halfcheetah-medium-v0'],
        'eevee': ['walker2d-random-v0',
                  'walker2d-medium-v0',
                  'walker2d-expert-v0'],
        'flareon': ['halfcheetah-random-v0',
                    'halfcheetah-medium-v0',
                    'halfcheetah-expert-v0',
                    'halfcheetah-medium-replay-v0',
                    'halfcheetah-medium-expert-v0',
                    'walker2d-random-v0',
                    'walker2d-medium-v0',
                    'walker2d-expert-v0',
                    'walker2d-medium-replay-v0',
                    'walker2d-medium-expert-v0'],
        'antmaze': ['antmaze-umaze-v0',
                    'antmaze-umaze-diverse-v0',
                    'antmaze-medium-play-v0',
                    'antmaze-medium-diverse-v0',
                    'antmaze-large-play-v0',
                    'antmaze-large-diverse-v0'],
    }
    ENVS = TOC[args.envset]

    if CLUSTER == 'baobab':
        # Define per-environment partitions map
        PEP = {k.split('-v')[0]: PARTITION for k in ENVS}
        # Define per-environment ntasks map
        PEC = {k.split('-v')[0]: NUM_WORKERS for k in ENVS}
        # Define per-environment timeouts map
        PET = {k.split('-v')[0]: TIMEOUT for k in ENVS}

else:
    raise NotImplementedError("benchmark not covered by the spawner.")
assert bool(TOC), "each benchmark must have a 'TOC' dictionary"

# If needed, create the list of demonstrations
if NEED_DEMOS:
    demo_dir = os.environ['DEMO_DIR']
    DEMOS = {k: os.path.join(demo_dir, k) for k in ENVS}
# If needed, create the list of datasets
if NEED_DSETS:
    d4rl_dir = os.environ['D4RL_DIR']
    DSETS = {k: os.path.join(d4rl_dir, k + '.h5') for k in ENVS}


def copy_and_add_seed(hpmap, seed):
    hpmap_ = deepcopy(hpmap)
    # Add the seed and edit the job uuid to only differ by the seed
    hpmap_.update({'seed': seed})
    # Enrich the uuid with extra information
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        gitsha = "gitSHA_{}".format(out.strip().decode('ascii'))
    except OSError:
        pass
    if NEED_DEMOS:
        hpmap_.update({'uuid': "{}.{}.{}.demos{}.seed{}".format(
            hpmap['uuid'],
            gitsha,
            hpmap['env_id'],
            str(hpmap['num_demos']).zfill(3),
            str(seed).zfill(2))}
        )
    else:
        hpmap_.update({'uuid': "{}.{}.{}.seed{}".format(
            hpmap['uuid'],
            gitsha,
            hpmap['env_id'],
            str(seed).zfill(2))}
        )
    return hpmap_


def copy_and_add_env(hpmap, env):
    hpmap_ = deepcopy(hpmap)
    # Add the env and demos or dataset
    hpmap_.update({'env_id': env})
    if NEED_DEMOS:
        hpmap_.update({'expert_path': DEMOS[env]})
    if NEED_DSETS:
        hpmap_.update({'dataset_path': DSETS[env]})
    return hpmap_


def copy_and_add_num_demos(hpmap, num_demos):
    hpmap_ = deepcopy(hpmap)
    # Add the num of demos
    hpmap_.update({'num_demos': num_demos})
    return hpmap_


def get_hps(sweep):
    """Return a list of maps of hyperparameters"""
    # Create a uuid to identify the current job
    uuid = create_uuid()
    # Assemble the hyperparameter map
    if sweep:
        # Random search
        hpmap = {
            # Primary
            'wandb_project': WANDB_PROJECT,

            # Generic
            'uuid': uuid,
            'cuda': CONFIG['parameters']['cuda'],
            'render': False,
            'record': CONFIG['logging'].get('record', False),
            'task': CONFIG['parameters']['task'],
            'algo': CONFIG['parameters']['algo'],

            # Training
            'save_frequency': int(float(CONFIG['parameters'].get('save_frequency', 400))),
            'num_timesteps': int(float(CONFIG['parameters'].get('num_timesteps', 2e7))),
            'training_steps_per_iter': CONFIG['parameters'].get('training_steps_per_iter', 2),
            'eval_steps_per_iter': CONFIG['parameters'].get('eval_steps_per_iter', 10),
            'eval_frequency': int(float(CONFIG['parameters'].get('eval_frequency', 10))),

            # Model
            'perception_stack': CONFIG['parameters']['perception_stack'],
            'layer_norm': CONFIG['parameters']['layer_norm'],

            # Optimization
            'actor_lr': float(np.random.choice([1e-4, 3e-4])),
            'critic_lr': float(np.random.choice([1e-4, 3e-4])),
            'with_scheduler': CONFIG['parameters']['with_scheduler'],
            'clip_norm': CONFIG['parameters']['clip_norm'],
            'wd_scale': float(np.random.choice([1e-4, 3e-4, 1e-3])),

            # Algorithm
            'rollout_len': np.random.choice([2, 5]),
            'batch_size': np.random.choice([32, 64, 128]),
            'gamma': np.random.choice([0.99, 0.995]),
            'mem_size': np.random.choice([10000, 50000, 100000]),
            'noise_type': np.random.choice(['"adaptive-param_0.2, normal_0.2"',
                                            '"adaptive-param_0.2, ou_0.2"',
                                            '"normal_0.2"',
                                            '"ou_0.2"']),
            'pn_adapt_frequency': CONFIG['parameters'].get('pn_adapt_frequency', 50),
            'polyak': np.random.choice([0.001, 0.005, 0.01]),
            'targ_up_freq': np.random.choice([10, 1000]),
            'n_step_returns': CONFIG['parameters'].get('n_step_returns', False),
            'lookahead': np.random.choice([5, 10, 20, 40, 60]),
            'ret_norm': CONFIG['parameters'].get('ret_norm', False),
            'popart': CONFIG['parameters'].get('popart', False),

            # TD3
            'clipped_double': CONFIG['parameters'].get('clipped_double', False),
            'targ_actor_smoothing': CONFIG['parameters'].get('targ_actor_smoothing', False),
            'td3_std': CONFIG['parameters'].get('td3_std', 0.2),
            'td3_c': CONFIG['parameters'].get('td3_c', 0.5),
            'actor_update_delay': np.random.choice([2, 3, 4]),

            # Prioritized replay
            'prioritized_replay': CONFIG['parameters'].get('prioritized_replay', False),
            'alpha': CONFIG['parameters'].get('alpha', 0.3),
            'beta': CONFIG['parameters'].get('beta', 1.),
            'ranked': CONFIG['parameters'].get('ranked', False),
            'unreal': CONFIG['parameters'].get('unreal', False),

            # Distributional RL
            'use_c51': CONFIG['parameters'].get('use_c51', False),
            'use_qr': CONFIG['parameters'].get('use_qr', False),
            'c51_num_atoms': CONFIG['parameters'].get('c51_num_atoms', 51),
            'c51_vmin': CONFIG['parameters'].get('c51_vmin', -10.),
            'c51_vmax': CONFIG['parameters'].get('c51_vmax', 10.),
            'num_tau': np.random.choice([100, 200]),

            # Offline RL
            'offline': CONFIG['parameters']['offline'],
            'use_expert_demos': CONFIG['parameters']['use_expert_demos'],
            'sub_rate': CONFIG['parameters']['sub_rate'],

            # SAC, BCQ, BEAR
            'state_dependent_std': CONFIG['parameters'].get('state_dependent_std', False),
            'vae_lr': CONFIG['parameters'].get('vae_lr', 1e-3),
            'use_adaptive_alpha': CONFIG['parameters'].get('use_adaptive_alpha', True),
            'alpha_lr': CONFIG['parameters'].get('alpha_lr', 1e-4),
            'init_temperature': CONFIG['parameters'].get('init_temperature', 0.1),
            'crit_targ_update_freq': CONFIG['parameters'].get('crit_targ_update_freq', 2),
        }
    else:
        # No search, fixed map
        hpmap = {
            # Primary
            'wandb_project': WANDB_PROJECT,

            # Generic
            'uuid': uuid,
            'cuda': CONFIG['parameters']['cuda'],
            'render': False,
            'record': CONFIG['logging'].get('record', False),
            'task': CONFIG['parameters']['task'],
            'algo': CONFIG['parameters']['algo'],

            # Training
            'save_frequency': int(float(CONFIG['parameters'].get('save_frequency', 400))),
            'num_timesteps': int(float(CONFIG['parameters'].get('num_timesteps', 2e7))),
            'training_steps_per_iter': CONFIG['parameters'].get('training_steps_per_iter', 2),
            'eval_steps_per_iter': CONFIG['parameters'].get('eval_steps_per_iter', 10),
            'eval_frequency': int(float(CONFIG['parameters'].get('eval_frequency', 10))),

            # Model
            'perception_stack': CONFIG['parameters']['perception_stack'],
            'layer_norm': CONFIG['parameters']['layer_norm'],

            # Optimization
            'actor_lr': float(CONFIG['parameters'].get('actor_lr', 1e-4)),
            'critic_lr': float(CONFIG['parameters'].get('critic_lr', 1e-4)),
            'with_scheduler': CONFIG['parameters']['with_scheduler'],
            'clip_norm': CONFIG['parameters']['clip_norm'],
            'wd_scale': float(CONFIG['parameters'].get('wd_scale', 3e-4)),

            # Algorithm
            'rollout_len': CONFIG['parameters'].get('rollout_len', 2),
            'batch_size': CONFIG['parameters'].get('batch_size', 128),
            'gamma': CONFIG['parameters'].get('gamma', 0.99),
            'mem_size': int(float(CONFIG['parameters'].get('mem_size', 100000))),
            'noise_type': CONFIG['parameters'].get('noise_type', 'none'),
            'pn_adapt_frequency': CONFIG['parameters'].get('pn_adapt_frequency', 50),
            'polyak': CONFIG['parameters'].get('polyak', 0.005),
            'targ_up_freq': CONFIG['parameters'].get('targ_up_freq', 100),
            'n_step_returns': CONFIG['parameters'].get('n_step_returns', False),
            'lookahead': CONFIG['parameters'].get('lookahead', 10),
            'ret_norm': CONFIG['parameters'].get('ret_norm', False),
            'popart': CONFIG['parameters'].get('popart', False),

            # TD3
            'clipped_double': CONFIG['parameters'].get('clipped_double', False),
            'targ_actor_smoothing': CONFIG['parameters'].get('targ_actor_smoothing', False),
            'td3_std': CONFIG['parameters'].get('td3_std', 0.2),
            'td3_c': CONFIG['parameters'].get('td3_c', 0.5),
            'actor_update_delay': CONFIG['parameters'].get('actor_update_delay', 2),

            # Prioritized replay
            'prioritized_replay': CONFIG['parameters'].get('prioritized_replay', False),
            'alpha': CONFIG['parameters'].get('alpha', 0.3),
            'beta': CONFIG['parameters'].get('beta', 1.),
            'ranked': CONFIG['parameters'].get('ranked', False),
            'unreal': CONFIG['parameters'].get('unreal', False),

            # Distributional RL
            'use_c51': CONFIG['parameters'].get('use_c51', False),
            'use_qr': CONFIG['parameters'].get('use_qr', False),
            'c51_num_atoms': CONFIG['parameters'].get('c51_num_atoms', 51),
            'c51_vmin': CONFIG['parameters'].get('c51_vmin', -10.),
            'c51_vmax': CONFIG['parameters'].get('c51_vmax', 10.),
            'num_tau': CONFIG['parameters'].get('num_tau', 200),

            # Offline RL
            'offline': CONFIG['parameters']['offline'],
            'use_expert_demos': CONFIG['parameters']['use_expert_demos'],
            'sub_rate': CONFIG['parameters']['sub_rate'],

            # SAC, BCQ, BEAR
            'state_dependent_std': CONFIG['parameters'].get('state_dependent_std', False),
            'vae_lr': CONFIG['parameters'].get('vae_lr', 1e-3),
            'use_adaptive_alpha': CONFIG['parameters'].get('use_adaptive_alpha', True),
            'alpha_lr': CONFIG['parameters'].get('alpha_lr', 1e-4),
            'init_temperature': CONFIG['parameters'].get('init_temperature', 0.1),
            'crit_targ_update_freq': CONFIG['parameters'].get('crit_targ_update_freq', 2),
        }

    # Duplicate for each environment
    hpmaps = [copy_and_add_env(hpmap, env)
              for env in ENVS]

    if NEED_DEMOS:
        # Duplicate for each number of demos
        hpmaps = [copy_and_add_num_demos(hpmap_, num_demos)
                  for hpmap_ in hpmaps
                  for num_demos in NUM_DEMOS]

    # Duplicate for each seed
    hpmaps = [copy_and_add_seed(hpmap_, seed)
              for hpmap_ in hpmaps
              for seed in range(NUM_SEEDS)]

    # Verify that the correct number of configs have been created
    assert len(hpmaps) == NUM_SEEDS * len(ENVS) * len(NUM_DEMOS)

    return hpmaps


def unroll_options(hpmap):
    """Transform the dictionary of hyperparameters into a string of bash options"""
    indent = 4 * ' '  # indents are defined as 4 spaces
    arguments = ""

    for k, v in hpmap.items():
        if k in BOOL_ARGS:
            if v is False:
                argument = "no-{}".format(k)
            else:
                argument = "{}".format(k)
        else:
            argument = "{}={}".format(k, v)

        arguments += "{}--{} \\\n".format(indent, argument)

    return arguments


def create_job_str(name, command, envkey):
    """Build the batch script that launches a job"""

    # Prepend python command with python binary path
    command = os.path.join(os.environ['CONDA_PREFIX'], "bin", command)

    if CLUSTER == 'baobab':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={jobname}\n'
                            '#SBATCH --partition={partition}\n'
                            '#SBATCH --ntasks={ntasks}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={timeout}\n'
                            '#SBATCH --mem=32000\n'
                            '#SBATCH --output=./out/run_%j.out\n'
                            '#SBATCH --constraint="V3|V4|V5|V6|V7"\n')
        if CONFIG['parameters']['cuda']:
            contraint = "COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
            bash_script_str += ('#SBATCH --gres=gpu:1\n'
                                '#SBATCH --constraint="{}"\n'.format(contraint))
        bash_script_str += ('\n')
        # Load modules
        bash_script_str += ('module load GCC/8.3.0 OpenMPI/3.1.4\n')
        if BENCH in ['dmc', 'd4rl']:
            bash_script_str += ('module load Mesa/19.2.1\n')
        if CONFIG['parameters']['cuda']:
            bash_script_str += ('module load CUDA\n')
        bash_script_str += ('\n')
        # Launch command
        bash_script_str += ('srun {command}')

        bash_script_str = bash_script_str.format(jobname=name,
                                                 partition=PEP[envkey],
                                                 ntasks=PEC[envkey],
                                                 timeout=PET[envkey],
                                                 command=command)

    elif CLUSTER == 'local':
        # Set header
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('# job name: {}\n\n')
        # Launch command
        bash_script_str += ('mpiexec -n {} {}')
        bash_script_str = bash_script_str.format(name,
                                                 CONFIG['resources']['num_workers'],
                                                 command)
    else:
        raise NotImplementedError("cluster selected is not covered.")

    return bash_script_str[:-2]  # remove the last `\` and `\n` tokens


def run(args):
    """Spawn jobs"""

    if args.wandb_upgrade:
        # Upgrade the wandb package
        logger.info(">>>>>>>>>>>>>>>>>>>> Upgrading wandb pip package")
        out = subprocess.check_output([sys.executable, '-m', 'pip', 'install', 'wandb', '--upgrade'])
        logger.info(out.decode("utf-8"))

    # Create directory for spawned jobs
    root = os.path.dirname(os.path.abspath(__file__))
    spawn_dir = os.path.join(root, 'spawn')
    os.makedirs(spawn_dir, exist_ok=True)
    if CLUSTER == 'local':
        tmux_dir = os.path.join(root, 'tmux')
        os.makedirs(tmux_dir, exist_ok=True)

    # Get the hyperparameter set(s)
    if args.sweep:
        hpmaps_ = [get_hps(sweep=True)
                   for _ in range(CONFIG['parameters']['num_trials'])]
        # Flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = get_hps(sweep=False)

    # Create associated task strings
    commands = ["python main.py \\\n{}".format(unroll_options(hpmap)) for hpmap in hpmaps]
    if not len(commands) == len(set(commands)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again (:")
    # Create the job maps
    names = ["{}.{}".format(TYPE, hpmap['uuid']) for i, hpmap in enumerate(hpmaps)]
    # Create environment keys for envionment-specific hyperparameter selection
    envkeys = [hpmap['env_id'].split('-v')[0] for hpmap in hpmaps]

    # Finally get all the required job strings
    jobs = [create_job_str(name, command, envkey)
            for name, command, envkey in zipsame(names, commands, envkeys)]

    # Spawn the jobs
    for i, (name, job) in enumerate(zipsame(names, jobs)):
        logger.info(">>>>>>>>>>>>>>>>>>>> Job #{} ready to submit. Config below.".format(i))
        logger.info(job + "\n")
        dirname = name.split('.')[1]
        full_dirname = os.path.join(spawn_dir, dirname)
        os.makedirs(full_dirname, exist_ok=True)
        job_name = os.path.join(full_dirname, "{}.sh".format(name))
        with open(job_name, 'w') as f:
            f.write(job)
        if args.call and not CLUSTER == 'local':
            # Spawn the job!
            stdout = subprocess.run(["sbatch", job_name]).stdout
            logger.info("[STDOUT]\n{}".format(stdout))
            logger.info(">>>>>>>>>>>>>>>>>>>> Job #{} submitted.".format(i))
    # Summarize the number of jobs spawned
    logger.info(">>>>>>>>>>>>>>>>>>>> {} jobs were spawned.".format(len(jobs)))

    if CLUSTER == 'local':
        dir_ = hpmaps[0]['uuid'].split('.')[0]  # arbitrarilly picked index 0
        session_name = "{}-{}seeds-{}".format(TYPE, str(NUM_SEEDS).zfill(2), dir_)
        yaml_content = {'session_name': session_name,
                        'windows': []}
        if NEED_DEMOS:
            yaml_content.update({'environment': {'DEMO_DIR': os.environ['DEMO_DIR']}})
        if NEED_DSETS:
            yaml_content.update({'environment': {'D4RL_DIR': os.environ['D4RL_DIR']}})
        for i, name in enumerate(names):
            executable = "{}.sh".format(name)
            pane = {'shell_command': ["source activate {}".format(CONDA),
                                      "chmod u+x spawn/{}/{}".format(dir_, executable),
                                      "spawn/{}/{}".format(dir_, executable)]}
            window = {'window_name': "job{}".format(str(i).zfill(2)),
                      'focus': False,
                      'panes': [pane]}
            yaml_content['windows'].append(window)
        # Dump the assembled tmux config into a yaml file
        job_config = os.path.join(tmux_dir, "{}.yaml".format(session_name))
        with open(job_config, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        if args.call:
            # Spawn all the jobs in the tmux session!
            stdout = subprocess.run(["tmuxp", "load", "-d", "{}".format(job_config)]).stdout
            logger.info("[STDOUT]\n{}".format(stdout))


if __name__ == "__main__":
    # Create (and optionally launch) the jobs!
    run(args)
