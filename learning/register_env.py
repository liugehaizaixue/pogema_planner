import numpy as np
from sample_factory.utils.typing import Env
from sample_factory.envs.env_utils import register_env
from argparse import Namespace
import gymnasium
from learning.epom_config import Environment, Experiment
from learning.grid_memory import GridMemoryWrapper
from learning.obs_memory import ObsMemoryWrapper
from pomapf_env.env import make_pomapf
from pomapf_env.wrappers import MatrixObservationWrapper


def create_pogema_env(cfg: Environment=None):
    env = make_pomapf(grid_config=cfg.grid_config)
    gm_radius = cfg.grid_memory_obs_radius
    memory_type = cfg.memory_type
    memory_length = cfg.memory_length
    env = GridMemoryWrapper(env, obs_radius=gm_radius if gm_radius else cfg.grid_config.obs_radius , memory_type=memory_type)
    env = MatrixObservationWrapper(env, memory_type=memory_type)
    env = ObsMemoryWrapper(env, obs_radius=gm_radius if gm_radius else cfg.grid_config.obs_radius, memory_length=memory_length)
    return env


def create_env(environment_cfg: Environment):
    env = create_pogema_env(environment_cfg)
    return env


class MultiEnv(gymnasium.Wrapper):
    def __init__(self, env_cfg: Environment):
        if env_cfg.target_num_agents is None:
            self.envs = [create_env(env_cfg)]
        else:
            assert env_cfg.target_num_agents % env_cfg.grid_config.num_agents == 0, \
                "Target num follower must be divisible by num agents"
            num_envs = env_cfg.target_num_agents // env_cfg.grid_config.num_agents
            self.envs = [create_env(env_cfg) for _ in range(num_envs)]

        super().__init__(self.envs[0])

    def step(self, actions):
        obs, rewards, dones, truncated, infos = [], [], [], [], []
        last_agents = 0
        for env in self.envs:
            env_num_agents = env.get_num_agents()
            action = actions[last_agents: last_agents + env_num_agents]
            last_agents = last_agents + env_num_agents
            o, r, d, t, i = env.step(action)
            obs += o
            rewards += r
            dones += d
            truncated += t
            infos += i
        return obs, rewards, dones, truncated, infos

    def reset(self, seed, **kwargs):
        obs = []
        for idx, env in enumerate(self.envs):
            inner_seed = seed + idx
            o, _ = env.reset(seed=inner_seed, **kwargs)
            obs += o
        return obs, {}

    def sample_actions(self):
        actions = []
        for env in self.envs:
            actions += list(env.sample_actions())
        return np.array(actions)

    @property
    def num_agents(self):
        return sum([env.get_num_agents() for env in self.envs])

    def render(self):
        for q in self.envs:
            q.render()


def make_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    p_config = Experiment(**vars(cfg))
    environment_config = p_config.environment

    if environment_config.agent_bins is not None and environment_config.target_num_agents is not None:
        if environment_config.env_id is None:
            num_agents = environment_config.agent_bins[0]
        else:
            num_agents = environment_config.agent_bins[environment_config.env_id % len(environment_config.agent_bins)]
        environment_config.grid_config.num_agents = num_agents

        return MultiEnv(environment_config)
    return create_env(environment_config)


class CustomEnv:
    def make_env(self, env_name, cfg, env_config, render_mode) -> Env:
        return make_env(env_name, cfg, env_config, render_mode)


def register_pogema_envs(env_name):
    env_factory = CustomEnv()
    register_env(env_name, env_factory.make_env)


def register_custom_components(env_name):
    register_pogema_envs(env_name)
