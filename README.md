# Placeholder Title

## Sanity-preserving conda environment creation trace

### AlgaeDICE
```bash
# tensorflow==2.0.0
# tensorflow-probability==0.8.0
# tf-agents-nightly==0.2.0.dev20191125
# tfp-nightly==0.9.0.dev20191125
```

### CQL
```bash

```

### BRAC
```bash
tensorflow==1.14.0
tensorflow-probability==0.7.0rc0
```

### BC
```bash

```

### BEAR
```bash

```

### AWR
```bash
tensorflow>=1.12.0,<2.0.0
```

### BCQ
```bash

```

### Continuous REM
```bash

```

### Mine
```bash
conda create -n pytorch-rl python=3.7
conda activate pytorch-rl
# >>>> PyTorch
pip install --upgrade pip
pip install --upgrade pytest pytest-instafail flake8 wrapt six tqdm pyyaml psutil cloudpickle
pip install --upgrade numpy pandas scipy scikit-learn h5py matplotlib
pip install --upgrade pyvips scikit-image
pip install --upgrade torch torchvision
conda install -y -c conda-forge pillow opencv pyglet pyopengl mpi4py cython patchelf
# >>>> wandb
conda install -y -c conda-forge watchdog
pip install moviepy imageio
pip install wandb
# >>>> MuJoCo
brew install gcc@8
cd && mkdir -p .mujoco && cd .mujoco
curl -O https://www.roboti.us/download/mujoco200_macos.zip
unzip mujoco200_macos.zip
mv mujoco200_macos mujoco200
# >>>> Gym + MuJoCo
cd ~/Code
git clone https://github.com/openai/mujoco-py.git
pip install -e mujoco-py
git clone https://github.com/openai/gym.git
pip install -e 'gym[all]'
```
