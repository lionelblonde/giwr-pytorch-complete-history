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


ENV_BUNDLES = {
    'mujoco': {
        'debug': ['Hopper-v3'],
        'eevee': ['InvertedPendulum-v2',
                  'InvertedDoublePendulum-v2'],
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
    },
    'd4rl': {
        'debug': ['hopper-expert-v0'],
        'debug_antmaze': ['antmaze-medium-play-v0'],
        'debug_adroit': ['pen-human-v0'],
        'debug_kitchen': ['kitchen-partial-v0'],
        'eevee': ['hopper-random-v0',
                  'hopper-medium-v0',
                  'hopper-expert-v0'],
        'glaceon': ['walker2d-random-v0',
                    'walker2d-medium-v0',
                    'walker2d-expert-v0'],
        'umbreon': ['halfcheetah-random-v0',
                    'halfcheetah-medium-v0',
                    'halfcheetah-expert-v0'],
        'mujoco': ['hopper-random-v0',
                   'hopper-medium-v0',
                   'hopper-expert-v0',
                   'hopper-medium-replay-v0',
                   'hopper-medium-expert-v0',
                   'halfcheetah-random-v0',
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
        'adroit': ['pen-human-v0',
                   'pen-cloned-v0',
                   'pen-expert-v0',
                   'hammer-human-v0',
                   'hammer-cloned-v0',
                   'hammer-expert-v0',
                   'relocate-human-v0',
                   'relocate-cloned-v0',
                   'relocate-expert-v0',
                   'door-human-v0',
                   'door-cloned-v0',
                   'door-expert-v0'],
        'kitchen': ['kitchen-partial-v0',
                    'kitchen-complete-v0',
                    'kitchen-mixed-v0'],
        'suite': ['hopper-random-v0',
                  'hopper-medium-v0',
                  'hopper-expert-v0',
                  'hopper-medium-replay-v0',
                  'hopper-medium-expert-v0',
                  'halfcheetah-random-v0',
                  'halfcheetah-medium-v0',
                  'halfcheetah-expert-v0',
                  'halfcheetah-medium-replay-v0',
                  'halfcheetah-medium-expert-v0',
                  'walker2d-random-v0',
                  'walker2d-medium-v0',
                  'walker2d-expert-v0',
                  'walker2d-medium-replay-v0',
                  'walker2d-medium-expert-v0',
                  'antmaze-umaze-v0',
                  'antmaze-umaze-diverse-v0',
                  'antmaze-medium-play-v0',
                  'antmaze-medium-diverse-v0',
                  'antmaze-large-play-v0',
                  'antmaze-large-diverse-v0',
                  'pen-human-v0',
                  'pen-cloned-v0',
                  'pen-expert-v0',
                  'hammer-human-v0',
                  'hammer-cloned-v0',
                  'hammer-expert-v0',
                  'relocate-human-v0',
                  'relocate-cloned-v0',
                  'relocate-expert-v0',
                  'door-human-v0',
                  'door-cloned-v0',
                  'door-expert-v0',
                  'kitchen-partial-v0',
                  'kitchen-complete-v0',
                  'kitchen-mixed-v0'],
    },
}

MEMORY = 16


class Spawner(object):

    def __init__(self, args):
        self.args = args

        # Retrieve config from filesystem
        self.config = yaml.safe_load(open(self.args.config))

        # Check if we need expert demos, or datasets, or neither
        self.need_demos = self.config['use_expert_demos']
        assert not self.need_demos or self.config['offline']
        if self.need_demos:
            self.num_demos = [int(i) for i in self.args.num_demos]
            self.need_datasets = False
        else:
            self.num_demos = [0]  # arbitrary, only used for dim checking
            # If we are not using demos but are in the offline setting, we need the datasets
            if self.config['offline']:
                self.need_datasets = True

        # Assemble wandb project name
        self.wandb_project = (self.config['logging']['wandb_project'].upper() + '-' +
                              self.args.deployment.upper())

        # Define spawn type
        self.type = 'sweep' if self.args.sweep else 'fixed'

        # Define the needed memory in GB
        self.memory = MEMORY

        # Write out the boolean arguments (using the 'boolean_flag' function)
        self.bool_args = ['cuda', 'render', 'record', 'layer_norm',
                          'prioritized_replay', 'ranked', 'unreal', 'n_step_returns', 'ret_norm', 'popart',
                          'clipped_double', 'targ_actor_smoothing', 'use_c51', 'use_qr',
                          'offline', 'use_expert_demos', 'state_dependent_std', 'use_adaptive_alpha',
                          'brac_use_adaptive_alpha_ent', 'brac_use_adaptive_alpha_div', 'brac_value_kl_pen',
                          'cql_deterministic_backup', 'cql_use_adaptive_alpha_ent',
                          'cql_use_adaptive_alpha_pri', 'cql_use_version_3', 'ptso_use_targ_for_u']

        if self.args.deployment == 'slurm':
            # Translate intuitive 'caliber' into actual duration and partition on the Baobab cluster
            calibers = dict(short='0-06:00:00', long='0-12:00:00',  # partition: 'shared-EL7'
                            verylong='2-00:00:00', veryverylong='4-00:00:00')  # partition: 'mono-EL7'
            self.duration = calibers[self.args.caliber]  # intented KeyError trigger if invalid caliber
            if 'verylong' in self.args.caliber:
                self.partition = 'mono-EL7'
            else:
                if self.config['resources']['cuda']:
                    self.partition = 'shared-gpu-EL7'
                else:
                    self.partition = 'shared-EL7'

        # Define the set of considered environments from the considered suite
        self.envs = ENV_BUNDLES[self.config['meta']['benchmark']][self.args.env_bundle]

        if self.need_demos:
            # Create the list of demonstrations associated with the environments
            demo_dir = os.environ['DEMO_DIR']
            self.demos = {k: os.path.join(demo_dir, k) for k in self.envs}

        if self.need_datasets:
            # Create the list of datasets associated with the environments
            assert self.config['meta']['benchmark'] == 'd4rl'
            d4rl_dir = os.environ['D4RL_DIR']
            self.datasets = {k: os.path.join(d4rl_dir, k + '.h5') for k in self.envs}

    def copy_and_add_seed(self, hpmap, seed):
        hpmap_ = deepcopy(hpmap)

        # Add the seed and edit the job uuid to only differ by the seed
        hpmap_.update({'seed': seed})

        # Enrich the uuid with extra information
        try:
            out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            gitsha = "gitSHA_{}".format(out.strip().decode('ascii'))
        except OSError:
            pass

        uuid = f"{hpmap['uuid']}.{gitsha}.{hpmap['env_id']}.{hpmap['algo']}_{self.args.num_workers}"
        if self.need_demos:
            uuid += f".demos{str(hpmap['num_demos']).zfill(3)}"
        uuid += f".seed{str(seed).zfill(2)}"

        hpmap_.update({'uuid': uuid})

        return hpmap_

    def copy_and_add_env(self, hpmap, env):
        hpmap_ = deepcopy(hpmap)
        # Add the env and demos or dataset
        hpmap_.update({'env_id': env})
        if self.need_demos:
            hpmap_.update({'expert_path': self.demos[env]})
        if self.need_datasets:
            hpmap_.update({'dataset_path': self.datasets[env]})
        return hpmap_

    def copy_and_add_num_demos(self, hpmap, num_demos):
        assert self.need_demos
        hpmap_ = deepcopy(hpmap)
        # Add the num of demos
        hpmap_.update({'num_demos': num_demos})
        return hpmap_

    def get_hps(self):
        """Return a list of maps of hyperparameters"""

        # Create a uuid to identify the current job
        uuid = create_uuid()

        # Assemble the hyperparameter map
        if self.args.sweep:
            # Random search
            hpmap = {
                'wandb_project': self.wandb_project,
                'uuid': uuid,
                'cuda': self.config['resources']['cuda'],
                'gpu_index': self.config['resources'].get('gpu_index', 0),
                'render': False,
                'record': self.config['logging'].get('record', False),
                'task': self.config['meta']['task'],
                'algo': self.config['meta']['algo'],

                # Training
                'save_frequency': int(float(self.config.get('save_frequency', 1e5))),
                'num_steps': int(float(self.config.get('num_steps', 5e5))),
                'training_steps_per_iter': self.config.get('training_steps_per_iter', 2),
                'eval_steps_per_iter': self.config.get('eval_steps_per_iter', 10),
                'eval_frequency': int(float(self.config.get('eval_frequency', 10))),

                # Model
                'perception_stack': self.config['perception_stack'],
                'layer_norm': self.config['layer_norm'],

                # Optimization
                'actor_lr': float(np.random.choice([1e-4, 3e-4])),
                'critic_lr': float(np.random.choice([3e-4, 1e-3])),
                'lr_schedule': np.random.choice(['constant', 'linear']),
                'clip_norm': self.config['clip_norm'],
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
                'pn_adapt_frequency': self.config.get('pn_adapt_frequency', 50),
                'polyak': np.random.choice([0.001, 0.005, 0.01]),
                'targ_up_freq': np.random.choice([10, 1000]),
                'n_step_returns': self.config.get('n_step_returns', False),
                'lookahead': np.random.choice([5, 10, 20, 40, 60]),
                'ret_norm': self.config.get('ret_norm', False),
                'popart': self.config.get('popart', False),

                # TD3
                'clipped_double': self.config.get('clipped_double', False),
                'ensemble_q_lambda': self.config.get('ensemble_q_lambda', 1.0),
                'targ_actor_smoothing': self.config.get('targ_actor_smoothing', False),
                'td3_std': self.config.get('td3_std', 0.2),
                'td3_c': self.config.get('td3_c', 0.5),
                'actor_update_delay': np.random.choice([2, 3, 4]),

                # Prioritized replay
                'prioritized_replay': self.config.get('prioritized_replay', False),
                'alpha': self.config.get('alpha', 0.3),
                'beta': self.config.get('beta', 1.),
                'ranked': self.config.get('ranked', False),
                'unreal': self.config.get('unreal', False),

                # Distributional RL
                'use_c51': self.config.get('use_c51', False),
                'use_qr': self.config.get('use_qr', False),
                'c51_num_atoms': self.config.get('c51_num_atoms', 51),
                'c51_vmin': self.config.get('c51_vmin', -10.),
                'c51_vmax': self.config.get('c51_vmax', 10.),
                'num_tau': np.random.choice([100, 200]),

                # Offline RL
                'offline': self.config['offline'],
                'use_expert_demos': self.config['use_expert_demos'],
                'sub_rate': self.config['sub_rate'],

                'state_dependent_std': self.config.get('state_dependent_std', False),
                'bcq_phi': self.config.get('bcq_phi', 0.05),
                'behavior_lr': self.config.get('behavior_lr', 1e-3),
                'use_adaptive_alpha': self.config.get('use_adaptive_alpha', False),
                'log_alpha_lr': self.config.get('log_alpha_lr', 1e-4),
                'init_temp_log_alpha': self.config.get('init_temp_log_alpha', 0.1),
                'crit_targ_update_freq': self.config.get('crit_targ_update_freq', 2),
                'warm_start': self.config.get('warm_start', 20000),
                'bear_mmd_kernel': self.config.get('bear_mmd_kernel', 'laplacian'),
                'bear_mmd_sigma': self.config.get('bear_mmd_sigma', 20.),
                'bear_mmd_epsilon': self.config.get('bear_mmd_epsilon', 0.05),
                'brac_use_adaptive_alpha_ent': self.config.get('brac_use_adaptive_alpha_ent', False),
                'brac_use_adaptive_alpha_div': self.config.get('brac_use_adaptive_alpha_div', False),
                'brac_init_temp_log_alpha_ent': self.config.get('brac_init_temp_log_alpha_ent', 0.),
                'brac_init_temp_log_alpha_div': self.config.get('brac_init_temp_log_alpha_div', 1.),
                'brac_value_kl_pen': self.config.get('brac_value_kl_pen', True),
                'cql_deterministic_backup': self.config.get('cql_deterministic_backup', True),
                'cql_use_adaptive_alpha_ent': self.config.get('cql_use_adaptive_alpha_ent', False),
                'cql_use_adaptive_alpha_pri': self.config.get('cql_use_adaptive_alpha_pri', False),
                'cql_init_temp_log_alpha_ent': self.config.get('cql_init_temp_log_alpha_ent', 0.),
                'cql_init_temp_log_alpha_pri': self.config.get('cql_init_temp_log_alpha_pri', 1.),
                'cql_targ_lower_bound': self.config.get('cql_targ_lower_bound', 1.),
                'cql_min_q_weight': self.config.get('cql_min_q_weight', 5.),
                'cql_state_inflate': self.config.get('cql_state_inflate', 10),
                'cql_use_version_3': self.config.get('cql_use_version_3', True),

                'ptso_u_scale': self.config.get('ptso_u_scale', 1.),
                'ptso_use_targ_for_u': self.config.get('ptso_use_targ_for_u', False),
                'ptso_num_mat_updates_per_iter': self.config.get('ptso_num_mat_updates_per_iter', 10),
            }
        else:
            # No search, fixed hyper-parameters
            hpmap = {
                'wandb_project': self.wandb_project,
                'uuid': uuid,
                'cuda': self.config['resources']['cuda'],
                'gpu_index': self.config['resources'].get('gpu_index', 0),
                'render': False,
                'record': self.config['logging'].get('record', False),
                'task': self.config['meta']['task'],
                'algo': self.config['meta']['algo'],

                # Training
                'save_frequency': int(float(self.config.get('save_frequency', 1e5))),
                'num_steps': int(float(self.config.get('num_steps', 5e5))),
                'training_steps_per_iter': self.config.get('training_steps_per_iter', 2),
                'eval_steps_per_iter': self.config.get('eval_steps_per_iter', 10),
                'eval_frequency': int(float(self.config.get('eval_frequency', 10))),

                # Model
                'perception_stack': self.config['perception_stack'],
                'layer_norm': self.config['layer_norm'],

                # Optimization
                'actor_lr': float(self.config.get('actor_lr', 1e-4)),
                'critic_lr': float(self.config.get('critic_lr', 1e-3)),
                'lr_schedule': self.config['lr_schedule'],
                'clip_norm': self.config['clip_norm'],
                'wd_scale': float(self.config.get('wd_scale', 3e-4)),

                # Algorithm
                'rollout_len': self.config.get('rollout_len', 2),
                'batch_size': self.config.get('batch_size', 128),
                'gamma': self.config.get('gamma', 0.99),
                'mem_size': int(float(self.config.get('mem_size', 100000))),
                'noise_type': self.config.get('noise_type', 'none'),
                'pn_adapt_frequency': self.config.get('pn_adapt_frequency', 50),
                'polyak': self.config.get('polyak', 0.005),
                'targ_up_freq': self.config.get('targ_up_freq', 100),
                'n_step_returns': self.config.get('n_step_returns', False),
                'lookahead': self.config.get('lookahead', 10),
                'ret_norm': self.config.get('ret_norm', False),
                'popart': self.config.get('popart', False),

                # TD3
                'clipped_double': self.config.get('clipped_double', False),
                'ensemble_q_lambda': self.config.get('ensemble_q_lambda', 1.0),
                'targ_actor_smoothing': self.config.get('targ_actor_smoothing', False),
                'td3_std': self.config.get('td3_std', 0.2),
                'td3_c': self.config.get('td3_c', 0.5),
                'actor_update_delay': self.config.get('actor_update_delay', 2),

                # Prioritized replay
                'prioritized_replay': self.config.get('prioritized_replay', False),
                'alpha': self.config.get('alpha', 0.3),
                'beta': self.config.get('beta', 1.),
                'ranked': self.config.get('ranked', False),
                'unreal': self.config.get('unreal', False),

                # Distributional RL
                'use_c51': self.config.get('use_c51', False),
                'use_qr': self.config.get('use_qr', False),
                'c51_num_atoms': self.config.get('c51_num_atoms', 51),
                'c51_vmin': self.config.get('c51_vmin', -10.),
                'c51_vmax': self.config.get('c51_vmax', 10.),
                'num_tau': self.config.get('num_tau', 200),

                # Offline RL
                'offline': self.config['offline'],
                'use_expert_demos': self.config['use_expert_demos'],
                'sub_rate': self.config['sub_rate'],

                'state_dependent_std': self.config.get('state_dependent_std', False),
                'bcq_phi': self.config.get('bcq_phi', 0.05),
                'behavior_lr': self.config.get('behavior_lr', 1e-3),
                'use_adaptive_alpha': self.config.get('use_adaptive_alpha', False),
                'log_alpha_lr': self.config.get('log_alpha_lr', 1e-4),
                'init_temp_log_alpha': self.config.get('init_temp_log_alpha', 0.1),
                'crit_targ_update_freq': self.config.get('crit_targ_update_freq', 2),
                'warm_start': self.config.get('warm_start', 20000),
                'bear_mmd_kernel': self.config.get('bear_mmd_kernel', 'laplacian'),
                'bear_mmd_sigma': self.config.get('bear_mmd_sigma', 20.),
                'bear_mmd_epsilon': self.config.get('bear_mmd_epsilon', 0.05),
                'brac_use_adaptive_alpha_ent': self.config.get('brac_use_adaptive_alpha_ent', False),
                'brac_use_adaptive_alpha_div': self.config.get('brac_use_adaptive_alpha_div', False),
                'brac_init_temp_log_alpha_ent': self.config.get('brac_init_temp_log_alpha_ent', 0.),
                'brac_init_temp_log_alpha_div': self.config.get('brac_init_temp_log_alpha_div', 0.3),
                'brac_value_kl_pen': self.config.get('brac_value_kl_pen', True),
                'cql_deterministic_backup': self.config.get('cql_deterministic_backup', True),
                'cql_use_adaptive_alpha_ent': self.config.get('cql_use_adaptive_alpha_ent', False),
                'cql_use_adaptive_alpha_pri': self.config.get('cql_use_adaptive_alpha_pri', False),
                'cql_init_temp_log_alpha_ent': self.config.get('cql_init_temp_log_alpha_ent', 0.),
                'cql_init_temp_log_alpha_pri': self.config.get('cql_init_temp_log_alpha_pri', 1.),
                'cql_targ_lower_bound': self.config.get('cql_targ_lower_bound', 1.),
                'cql_min_q_weight': self.config.get('cql_min_q_weight', 5.),
                'cql_state_inflate': self.config.get('cql_state_inflate', 10),
                'cql_use_version_3': self.config.get('cql_use_version_3', True),

                'ptso_u_scale': self.config.get('ptso_u_scale', 1.),
                'ptso_use_targ_for_u': self.config.get('ptso_use_targ_for_u', False),
                'ptso_num_mat_updates_per_iter': self.config.get('ptso_num_mat_updates_per_iter', 10),
            }

        # Duplicate for each environment
        hpmaps = [self.copy_and_add_env(hpmap, env)
                  for env in self.envs]

        if self.need_demos:
            # Duplicate for each number of demos
            hpmaps = [self.copy_and_add_num_demos(hpmap_, num_demos)
                      for hpmap_ in hpmaps
                      for num_demos in self.num_demos]

        # Duplicate for each seed
        hpmaps = [self.copy_and_add_seed(hpmap_, seed)
                  for hpmap_ in hpmaps
                  for seed in range(self.args.num_seeds)]

        # Verify that the correct number of configs have been created
        assert len(hpmaps) == self.args.num_seeds * len(self.envs) * len(self.num_demos)

        return hpmaps

    def unroll_options(self, hpmap):
        """Transform the dictionary of hyperparameters into a string of bash options"""
        indent = 4 * ' '  # choice: indents are defined as 4 spaces
        arguments = ""

        for k, v in hpmap.items():
            if k in self.bool_args:
                if v is False:
                    argument = f"no-{k}"
                else:
                    argument = f"{k}"
            else:
                argument = f"{k}={v}"

            arguments += f"{indent}--{argument} \\\n"

        return arguments

    def create_job_str(self, name, command):
        """Build the batch script that launches a job"""

        # Prepend python command with python binary path
        command = os.path.join(os.environ['CONDA_PREFIX'], "bin", command)

        if self.args.deployment == 'slurm':
            os.makedirs("./out", exist_ok=True)
            # Set sbatch config
            bash_script_str = ('#!/usr/bin/env bash\n\n')
            bash_script_str += (f"#SBATCH --job-name={name}\n"
                                f"#SBATCH --partition={self.partition}\n"
                                f"#SBATCH --ntasks={self.args.num_workers}\n"
                                "#SBATCH --cpus-per-task=1\n"
                                f"#SBATCH --time={self.duration}\n"
                                f"#SBATCH --mem={self.memory}000\n"
                                "#SBATCH --output=./out/run_%j.out\n"
                                '#SBATCH --constraint="V3|V4|V5|V6|V7"\n')  # single quote to escape
            if self.config['resources']['cuda']:
                contraint = "COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
                bash_script_str += ("#SBATCH --gres=gpu:1\n"
                                    f'#SBATCH --constraint="{contraint}"\n')  # single quote to escape
            bash_script_str += ('\n')
            # Load modules
            bash_script_str += ("module load GCC/8.3.0 OpenMPI/3.1.4\n")
            if self.config['meta']['benchmark'] == 'd4rl':  # legacy comment: needed for dmc too
                bash_script_str += ("module load Mesa/19.2.1\n")
            if self.config['resources']['cuda']:
                bash_script_str += ("module load CUDA\n")
            bash_script_str += ('\n')
            # Launch command
            bash_script_str += (f"srun {command}")

        elif self.args.deployment == 'tmux':
            # Set header
            bash_script_str = ("#!/usr/bin/env bash\n\n")
            bash_script_str += (f"# job name: {name}\n\n")
            # Launch command
            bash_script_str += (f"mpiexec -n {self.args.num_workers} {command}")

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

    # Create a spawner object
    spawner = Spawner(args)

    # Create directory for spawned jobs
    root = os.path.dirname(os.path.abspath(__file__))
    spawn_dir = os.path.join(root, 'spawn')
    os.makedirs(spawn_dir, exist_ok=True)
    if args.deployment == 'tmux':
        tmux_dir = os.path.join(root, 'tmux')
        os.makedirs(tmux_dir, exist_ok=True)

    # Get the hyperparameter set(s)
    if args.sweep:
        hpmaps_ = [spawner.get_hps()
                   for _ in range(spawner.config['num_trials'])]
        # Flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = spawner.get_hps()

    # Create associated task strings
    commands = ["python main.py \\\n{}".format(spawner.unroll_options(hpmap)) for hpmap in hpmaps]
    if not len(commands) == len(set(commands)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again (:")
    # Create the job maps
    names = [f"{spawner.type}.{hpmap['uuid']}" for i, hpmap in enumerate(hpmaps)]

    # Finally get all the required job strings
    jobs = [spawner.create_job_str(name, command)
            for name, command in zipsame(names, commands)]

    # Spawn the jobs
    for i, (name, job) in enumerate(zipsame(names, jobs)):
        logger.info(f">>>>>>>>>>>>>>>>>>>> Job #{i} ready to submit. Config below.")
        logger.info(job + "\n")
        dirname = name.split('.')[1]
        full_dirname = os.path.join(spawn_dir, dirname)
        os.makedirs(full_dirname, exist_ok=True)
        job_name = os.path.join(full_dirname, f"{name}.sh")
        with open(job_name, 'w') as f:
            f.write(job)
        if args.deploy_now and not args.deployment == 'tmux':
            # Spawn the job!
            stdout = subprocess.run(["sbatch", job_name]).stdout
            logger.info(f"[STDOUT]\n{stdout}")
            logger.info(f">>>>>>>>>>>>>>>>>>>> Job #{i} submitted.")
    # Summarize the number of jobs spawned
    logger.info(f">>>>>>>>>>>>>>>>>>>> {len(jobs)} jobs were spawned.")

    if args.deployment == 'tmux':
        dir_ = hpmaps[0]['uuid'].split('.')[0]  # arbitrarilly picked index 0
        session_name = f"{spawner.type}-{str(args.num_seeds).zfill(2)}seeds-{dir_}"
        yaml_content = {'session_name': session_name,
                        'windows': []}
        if spawner.need_demos:
            yaml_content.update({'environment': {'DEMO_DIR': os.environ['DEMO_DIR']}})
        if spawner.need_datasets:
            yaml_content.update({'environment': {'D4RL_DIR': os.environ['D4RL_DIR']}})
        for i, name in enumerate(names):
            executable = f"{name}.sh"
            pane = {'shell_command': [f"source activate {args.conda_env}",
                                      f"chmod u+x spawn/{dir_}/{executable}",
                                      f"spawn/{dir_}/{executable}"]}
            window = {'window_name': f"job{str(i).zfill(2)}",
                      'focus': False,
                      'panes': [pane]}
            yaml_content['windows'].append(window)
        # Dump the assembled tmux config into a yaml file
        job_config = os.path.join(tmux_dir, f"{session_name}.yaml")
        with open(job_config, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        if args.deploy_now:
            # Spawn all the jobs in the tmux session!
            stdout = subprocess.run(["tmuxp", "load", "-d", job_config]).stdout
            logger.info(f"[STDOUT]\n{stdout}")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Job Spawner")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--conda_env', type=str, default=None)
    parser.add_argument('--env_bundle', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--deployment', type=str, choices=['tmux', 'slurm'], default='tmux', help='deploy how?')
    parser.add_argument('--num_seeds', type=int, default=None)
    parser.add_argument('--caliber', type=str, default=None)
    boolean_flag(parser, 'deploy_now', default=True, help="deploy immediately?")
    boolean_flag(parser, 'sweep', default=False, help="hp search?")
    boolean_flag(parser, 'wandb_upgrade', default=True, help="upgrade wandb?")
    parser.add_argument('--num_demos', '--list', nargs='+', type=str, default=None)
    args = parser.parse_args()

    # Create (and optionally deploy) the jobs
    run(args)
