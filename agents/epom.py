import json
from copy import deepcopy
from os.path import join
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch

from pydantic import Extra
from collections import OrderedDict
import numpy as np

from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from agents.utils_agents import AlgoBase, run_algorithm
from learning.epom_config import Environment
from learning.epom_config import Experiment

import os
from argparse import Namespace
from learning.grid_memory import MultipleGridMemory
from learning.obs_memory import MultipleObsMemory
from pomapf_env.wrappers import MatrixObservationWrapper

from learning.register_env import register_custom_components
from learning.register_training_utils import register_custom_model


class EpomConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['EPOM'] = 'EPOM'
    path_to_weights: str = "weights/epom"


class EPOM:
    def __init__(self, algo_cfg):
        self.algo_cfg: EpomConfig = algo_cfg
        register_custom_model()
        path = algo_cfg.path_to_weights
        device = algo_cfg.device

        self.path = path
        self.env = None
        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            flat_config = json.load(f)
            self.exp = Experiment(**flat_config)
            flat_config = Namespace(**flat_config)

        env_name = self.exp.environment.env
        register_custom_components(env_name)
        config = flat_config
        config.num_envs = 1
        env = make_env_func_batched(config, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
        actor_critic = create_actor_critic(config, env.observation_space, env.action_space)
        actor_critic.eval()
        env.close()

        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        self.device = device

        actor_critic.model_to_device(device)
        
        name_prefix = dict(latest="checkpoint", best="best")['latest']
        # policy_index = 0 if 'policy_index' not in flat_config else flat_config.policy_index
        checkpoints = Learner.get_checkpoints(self.path, f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = config

        self.rnn_states = None
        self.mgm = MultipleGridMemory()
        # self.mobsm = MultipleObsMemory()
        self._step = 0

    def after_reset(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.mgm.clear()
        self._step = 0

    def get_additional_info(self):
        result = {"rl_used": 1.0, }
        return result

    def get_name(self):
        return Path(self.path).name

    def act(self, observations, rewards=None, dones=None, infos=None):
        observations = deepcopy(observations)
        self.rnn_states = torch.zeros([len(observations), get_rnn_size(self.cfg)], dtype=torch.float32,
                                      device=self.device) if self.rnn_states is None else self.rnn_states
        
        env_cfg: Environment = Environment(**self.cfg.environment)
        self.mgm.update(observations)
        gm_radius = env_cfg.grid_memory_obs_radius
        self.mgm.modify_observation(observations, obs_radius=gm_radius if gm_radius else env_cfg.grid_config.obs_radius)
        observations = MatrixObservationWrapper.to_matrix(observations)
        # self.mobsm.update(observations)
        # observations_with_memory = self.mobsm.get_observations_with_memory(observations)

        with torch.no_grad():
            obs_torch = AttrDict(self.transform_dict_observations(observations_with_memory))
            normalized_obs = prepare_and_normalize_obs(self.ppo, obs_torch)
            policy_outputs = self.ppo(normalized_obs, self.rnn_states)
            self.rnn_states = policy_outputs['new_rnn_states']
            actions = policy_outputs['actions'].cpu().numpy()

        self._step += 1
        result = actions
        return result

    def clear_hidden(self, agent_idx):
        if self.rnn_states is not None:
            self.rnn_states[agent_idx] = torch.zeros([get_rnn_size(self.cfg)], dtype=torch.float32,
                                                     device=self.device)

    def after_step(self, dones):
        for agent_idx, done_flag in enumerate(dones):
            if done_flag:
                self.clear_hidden(agent_idx)

        if all(dones):
            self.rnn_states = None
            self.mgm.clear()

    @staticmethod
    def transform_dict_observations(observations):
        """Transform list of dict observations into a dict of lists."""
        obs_dict = dict()
        if isinstance(observations[0], (dict, OrderedDict)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            # handle flat observations also as dict
            obs_dict['obs'] = observations

        for key, x in obs_dict.items():
            obs_dict[key] = np.stack(x)

        return obs_dict


def example_epom(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64, main_dir='./',
                 animate=False):
    algo = EPOM(EpomConfig(path_to_weights=str(main_dir / Path('weights/epom'))))
    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate)


if __name__ == '__main__':
    print(example_epom(main_dir='../'))
