import gymnasium
import numpy as np
from gymnasium.spaces import Box
from pogema.grid import Grid


class ObsMemory:
    def __init__(self, memory_length = 10):
        self.memory_length = memory_length
        self.memory = None

    def update(self, observation):
        if self.memory is None or len(self.memory) != self.memory_length:
            self.memory = [observation for _ in range(self.memory_length)] # 如果记忆为空，则前几帧认为与当前帧相同，还是应该为空？

        self.memory.append(observation)
        self.memory.pop(0)

    def convert_to_dict(self, memory):
        result = {}
        for obs in memory:
            for key, value in obs.items():
                if key not in result:
                    result[key] = [value]
                else:
                    result[key].append(value)
        for k , v in result.items():
            result[k] = np.array(v)
        return result

    def get_observation(self):
        dict_obs = self.convert_to_dict(self.memory)
        return dict_obs


class MultipleObsMemory:
    def __init__(self, memory_length = 10):
        self.memories = None
        self.memory_length = memory_length

    def update(self, observations): #更改memory中的obstacles
        if self.memories is None or len(self.memories) != len(observations):
            self.memories = [ObsMemory(self.memory_length) for _ in range(len(observations))]

        for agent_idx, obs in enumerate(observations):
            self.memories[agent_idx].update(obs)


    def get_observations_with_memory(self, observations):
        return [self.memories[agent_idx].get_observation() for agent_idx, obs in enumerate(observations)] 

    def clear(self):
        self.memories = None


class ObsMemoryWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_radius, memory_length, display_directions=False):
        super().__init__(env)
        self.obs_radius = obs_radius
        self.memory_length = memory_length
        k = memory_length
        size = self.obs_radius * 2 + 1
        if display_directions:
            max_agents_value = 4.0
        else:
            max_agents_value = 1.0
        self.observation_space = gymnasium.spaces.Dict(
                obs=gymnasium.spaces.Box(-1.0, max_agents_value, shape=(k, 3, size, size)),
                xy=Box(low=-1024, high=1024, shape=(k, 2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(k, 2,), dtype=int),
                direction=Box(low=-1, high=1, shape=(k, 2,), dtype=int),
            )


        self.mobsm = MultipleObsMemory(self.memory_length)

    def observation(self, observations):
        self.mobsm.update(observations)
        observations_with_memory = self.mobsm.get_observations_with_memory(observations)
        return observations_with_memory

    def reset(self,seed=None, **kwargs):
        self.mobsm.clear()
        obs , infos = self.env.reset(seed=None, **kwargs)
        return self.observation(obs) , infos
