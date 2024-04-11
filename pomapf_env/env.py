from pogema import pogema_v0
from pogema.animation import AnimationMonitor, AnimationConfig
from pogema.integrations.sample_factory import AutoResetWrapper

from pomapf_env.wrappers import RewardShaping, MultiMapWrapper


def make_pomapf(grid_config):
    grid_config.auto_reset = False
    env = pogema_v0(grid_config)
    env = RewardShaping(env)

    env = MultiMapWrapper(env)
    env = AutoResetWrapper(env)
    return env
