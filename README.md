# Offline RL Baselines and Generalized Importance-Weighted Regression (GIWR)

PyTorch implementation of our work:
"Optimality Inductive Biases and Agnostic Guidelines for Offline Reinforcement Learning".
(Name of previous version:
"Where is the Grass Greener? Revisiting Generalized Policy Iteration for Offline Reinforcement Learning".)

[arXiv link](https://arxiv.org/abs/2107.01407)

## Contents

The repository covers the offline reinforcement learning algorithms
[BCQ](http://arxiv.org/abs/1812.02900),
[BEAR](http://arxiv.org/abs/1906.00949),
[BRAC](http://arxiv.org/abs/1911.11361),
[AWR](https://arxiv.org/abs/1910.00177),
[CRR](http://arxiv.org/abs/2006.15134), and
[CQL](http://arxiv.org/abs/2006.04779)
It also provides implementations for [SAC](http://arxiv.org/abs/1801.01290) and
for [D4PG](https://openreview.net/pdf?id=SyZipzbCb)
that were originally introduced as online methods, but here used offline.
The repository also provised an implementation of behavioral cloning (BC), the supervised learning approach
to imitation learning.
These are referred to as the offline *baselines* in [our paper](https://arxiv.org/abs/2107.01407).

The algorithms are implemented in PyTorch, and set to be trained and evaluated in the
[`d4rl`](https://github.com/rail-berkeley/d4rl) suite of environments and associated offline datasets
(installation instructions below).
Details about the D4RL offline RL benchmark are provided by the authors in
[the D4RL companion paper](http://arxiv.org/abs/2004.07219).

## Dependencies

### OS

Make sure you have [GLFW](https://www.glfw.org) and [Open MPI](https://www.open-mpi.org) installed on your system:
- if you are using macOS, run:
```bash
brew install open-mpi glfw3
```
- if you are using Ubuntu, run:
```bash
sudo apt -y install libopenmpi-dev libglfw3
```

### Python

Create a virtual enviroment for Python development using
[Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda)
or [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda):
- Create a conda environment for Python 3.7 called 'myenv', activate it, and upgrade `pip`:
```bash
conda create -n myenv python=3.7
conda activate myenv
# Once in the conda environment, upgrade the pip binary it uses to the latest
pip install --upgrade pip
```
- Install various core Python libraries:
```bash
# EITHER with versions that were used for this release
pip install pytest==5.2.1 pytest-instafail==0.4.1 flake8==3.7.9 wrapt==1.11.2 pillow==6.2.1 six==1.15.0 tqdm==4.36.1 pyyaml==5.1.2 psutil==5.6.3 cloudpickle==1.2.2 tmuxp==1.5.4 lockfile==0.12.2 numpy==1.17.4 pandas==0.25.2 scipy==1.3.1 scikit-learn==0.21.3 h5py==2.10.0 matplotlib==3.1.1 seaborn==0.9.0 pyvips==2.1.8 scikit-image==0.16.2 torch==1.6.0 torchvision==0.7.0
conda install -y -c conda-forge opencv=3.4.7 pyglet=1.3.2 pyopengl=3.1.5 mpi4py=3.0.2 cython=0.29.13 watchdog=0.9.0
pip install moviepy==1.0.1 imageio==2.6.1 wandb==0.10.10
# OR without versions (pulls the latest versions for each of these releases)
pip install pytest pytest-instafail flake8 wrapt pillow six tqdm pyyaml psutil cloudpickle tmuxp lockfile numpy pandas scipy scikit-learn h5py matplotlib seaborn pyvips scikit-image torch torchvision
conda install -y -c conda-forge opencv pyglet pyopengl mpi4py cython watchdog
pip install moviepy imageio wandb
```
- Install MuJoCo following the instructions laid out at
[`https://github.com/openai/mujoco-py#install-mujoco`](https://github.com/openai/mujoco-py#install-mujoco)
- Build [`mujoco-py`](https://github.com/openai/mujoco-py.git) from source in editable mode:
```bash
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
```
- Build [`gym`](https://github.com/openai/gym.git) from source in editable mode
(*cf.* [`http://gym.openai.com/docs/#installation`](http://gym.openai.com/docs/#installation)):
```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e ".[all]"
```
- Install [`d4rl`](https://github.com/rail-berkeley/d4rl.git) from source in editable mode
(*cf.* [`https://github.com/rail-berkeley/d4rl#setup`](https://github.com/rail-berkeley/d4rl#setup)):
```bash
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
```

### Offline Datasets

Download the [`d4rl`](https://github.com/rail-berkeley/d4rl) datasets and make them accessible:
- Go to this project root directory;
- Write the desired destination folder (to download the datasets to) in the file `dl_d4rl_datasets.py`;
- Download the datasets with the command: `python dl_d4rl_datasets.py`;
- Create the environment variable: `export D4RL_DIR=/where/you/downloaded/the/datasets`.

## Running Experiments
While one can launch any job via `main.py`, it is advised to use `spawner.py`,
designed to spawn a swarm of experiments over multiple seeds and environments in one command.
To get its usage description, type `python spawner.py -h`.
```bash
usage: spawner.py [-h] [--config CONFIG] [--conda_env CONDA_ENV]
                  [--env_bundle ENV_BUNDLE] [--num_workers NUM_WORKERS]
                  [--deployment {tmux,slurm,slurm2}] [--num_seeds NUM_SEEDS]
                  [--caliber CALIBER] [--deploy_now] [--no-deploy_now]
                  [--sweep] [--no-sweep] [--wandb_upgrade]
                  [--no-wandb_upgrade] [--debug] [--no-debug] [--wandb_dryrun]
                  [--no-wandb_dryrun] [--debug_lvl DEBUG_LVL]

Job Spawner

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG
  --conda_env CONDA_ENV
  --env_bundle ENV_BUNDLE
  --num_workers NUM_WORKERS
  --deployment {tmux,slurm,slurm2}
                        deploy how?
  --num_seeds NUM_SEEDS
  --caliber CALIBER
  --deploy_now          deploy immediately?
  --no-deploy_now
  --sweep               hp search?
  --no-sweep
  --wandb_upgrade       upgrade wandb?
  --no-wandb_upgrade
  --debug               toggle debug/verbose mode in spawner
  --no-debug
  --wandb_dryrun        toggle wandb offline mode
  --no-wandb_dryrun
  --debug_lvl DEBUG_LVL
                        set the debug level for the spawned runs
```

Here is an example:
```bash
python spawner.py --config tasks/train_d4rl_bcq.yaml --env_bundle debug --wandb_upgrade --no-sweep --deploy_now --caliber short --num_workers 2 --num_seeds 3 --deployment tmux --conda_env myenv --wandb_dryrun --debug_lvl 2
```
Check the argument parser in `spawner.py` to know what each of these arguments mean,
and how to adapt them to your needs.
