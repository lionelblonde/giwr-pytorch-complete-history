import os

import gym

import logger
import environments

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import d4rl  # noqa, import required to register environments


def get_benchmark(env_id):
    """Verify that the specified env is amongst the admissible ones"""
    for k, v in environments.BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    return benchmark


def make_env(env_id, seed):
    """Create an environment"""
    benchmark = get_benchmark(env_id)

    if benchmark == 'mujoco':
        # Remove the lockfile if it exists
        lockfile = os.path.join(
            os.environ['CONDA_PREFIX'],
            "lib",
            "python3.7",
            "site-packages",
            "mujoco_py",
            "generated",
            "mujocopy-buildlock.lock",
        )
        try:
            os.remove(lockfile)
            logger.info("[WARN] removed mujoco lockfile")
        except OSError:
            pass

    env = gym.make(env_id)
    env.seed(seed)

    return env
