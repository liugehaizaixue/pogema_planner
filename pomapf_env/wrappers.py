import re
from copy import deepcopy
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict
import numpy as np
from numpy import float32
from pogema import GridConfig
from pomapf_env.custom_maps import MAPS_REGISTRY
from learning.APF import calculate_apf
from learning.path_matrix import calculate_path

class RewardShaping(gymnasium.Wrapper):
    def __init__(self, env, density_reward = False):
        super().__init__(env)
        self._previous_loc = None
        self.density_reward = density_reward

    def step(self, action):
        observations, rewards, dones, truncated, infos = self.env.step(action)
        for agent_idx in range(self.get_num_agents()):
            # [TODO] ? 是否需要判断agent已完成任务
            reward = rewards[agent_idx]
            reward -= 0.0001
            if action[agent_idx] != 0:
                if tuple(self._previous_loc[agent_idx]) == (*observations[agent_idx]['xy'] , observations[agent_idx]['direction']):
                    reward -= 0.0002
            
            if self.density_reward: 
                reward += self.get_density_reward(observations[agent_idx])

            rewards[agent_idx] = reward
            self._previous_loc[agent_idx] = [*observations[agent_idx]['xy'] , observations[agent_idx]['direction']] 


        return observations, rewards, dones, truncated, infos

    def reset(self,seed=None, **kwargs):
        observation , infos = self.env.reset(seed=seed, **kwargs)
        self._previous_loc = [[0, 0, obs['direction'] ] for obs in observation]

        return observation, infos
    
    def get_density_reward(self, observation):
        # [TODO]
        # Reward based on the density of agents
        agents = np.sum( observation['agents'] == 1) # 至少为1，因为有自己
        obstacles = np.sum( observation['obstacles'] == 1) # 至少为0
        all_size = np.sum( observation['agents'] != -1) # 至少为2 ，不可能为1
        # free_size = np.sum( observation['obstacles'] == 0) # 最少为1，即面前只有一个障碍物，此时agents一定是1
        # if free_size == 1: 
        #     density = 0
        # else:
        #     density = (agents-1) / (free_size - 1)  # 考虑 障碍物密度 
        density = (obstacles + agents-1) / (all_size - 1)  # 不考虑 障碍物密度 
        reward = -0.0001 * density
        return reward
            



class MultiMapWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name

        if pattern:
            for map_name in MAPS_REGISTRY:
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def step(self, action):
        observations, rewards, done, truncated, info = self.env.step(action)
        cfg = self.grid_config
        if cfg.map_name:
            for agent_idx in range(self.get_num_agents()):
                if 'episode_extra_stats' in info[agent_idx]:
                    for key, value in list(info[agent_idx]['episode_extra_stats'].items()):
                        if key == 'Done':
                            continue
                        info[agent_idx]['episode_extra_stats'][f'{key}-{cfg.map_name.split("-")[0]}'] = value
        return observations, rewards, done, truncated, info

    def reset(self,seed=None, **kwargs):
        if self._configs is not None and len(self._configs) >= 1:
            cfg = deepcopy(self._configs[self._rnd.integers(0, len(self._configs))])
            self.env.unwrapped.grid_config = cfg
        obs , infos  = self.env.reset(seed=seed, **kwargs)
        return obs , infos


class MatrixObservationWrapper(ObservationWrapper):
    def __init__(self, env, memory_type="default", use_apf=False ):
        super().__init__(env)
        self.use_apf = use_apf
        if use_apf:
            obs_ch = 4 # [TODO]
        else:
            obs_ch = 3
        # full_size = self.config.obs_radius * 2 + 1
        max_agents_value = 1.0
        full_size = self.env.observation_space['obstacles'].shape[0]
        if memory_type == "default":
            self.observation_space = gymnasium.spaces.Dict(
                obs=gymnasium.spaces.Box(0.0, max_agents_value, shape=(obs_ch, full_size, full_size)),
                xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                direction=Box(low=-1, high=1, shape=(2,), dtype=int),
            )
        elif memory_type == "plus":
            self.observation_space = gymnasium.spaces.Dict(
                obs=gymnasium.spaces.Box(-1.0, max_agents_value, shape=(obs_ch, full_size, full_size)),
                xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                direction=Box(low=-1, high=1, shape=(2,), dtype=int),
            )
        elif memory_type == "max":
            self.observation_space = gymnasium.spaces.Dict(
                obs=gymnasium.spaces.Box(-1.0, 4.0, shape=(obs_ch, full_size, full_size)),
                xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                direction=Box(low=-1, high=1, shape=(2,), dtype=int),
            )

    @staticmethod
    def get_square_target(x, y, tx, ty, obs_radius):
        """ 此处会将目标点，放在矩阵中，
            若目标点位于范围外，则将其放在矩阵边界处
        """
        full_size = obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        dx, dy = x - tx, y - ty

        dx = min(dx, obs_radius) if dx >= 0 else max(dx, -obs_radius)
        dy = min(dy, obs_radius) if dy >= 0 else max(dy, -obs_radius)
        result[obs_radius - dx, obs_radius - dy] = 1
        return result

    @staticmethod
    def to_matrix(observations, use_apf=False):
        result = []
        if observations:
            obs_radius = observations[0]['obstacles'].shape[0] // 2
            for agent_idx, obs in enumerate(observations):   
                square_target = MatrixObservationWrapper.get_square_target(*obs['xy'], *obs['target_xy'], obs_radius)

                result.append(
                    {"obs": np.concatenate([obs['obstacles'][None], obs['agents'][None], square_target[None]]).astype(float32),
                    "xy": np.array(obs['xy'], dtype=float32),
                    "target_xy": np.array(obs['target_xy'], dtype=float32),
                    "direction": np.array(obs['direction'], dtype=float32),
                    })
                
                if use_apf: # [TODO]
                    matrix_path = calculate_path(obs['obstacles'], square_target)
                    result[-1]['obs'] = np.concatenate([result[-1]['obs'], matrix_path[None]])
        return result

    def observation(self, observation):
        result = self.to_matrix(observation,self.use_apf)
        return result
