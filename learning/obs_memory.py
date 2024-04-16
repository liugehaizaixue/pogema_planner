import gymnasium
import numpy as np
from gymnasium.spaces import Box
from pogema.grid import Grid


class ObsMemory:
    def __init__(self, memory_duration = 10):
        self.memory_duration = memory_duration
        self.memory = None

    def update(self, observation):
        if self.memory is None or len(self.memory) != self.memory_duration:
            self.memory = [observation for _ in range(self.memory_duration)] # 如果记忆为空，则前几帧认为与当前帧相同，还是应该为空？

        self.memory.append(observation)
        self.memory.pop(0)

    def get_observation(self):
        return self.memory


class MultipleObsMemory:
    def __init__(self):
        self.memories = None

    def update(self, observations): #更改memory中的obstacles
        if self.memories is None or len(self.memories) != len(observations):
            self.memories = [ObsMemory() for _ in range(len(observations))]

        for agent_idx, obs in enumerate(observations):
            self.memories[agent_idx].update(obs)


    def get_observations_with_memory(self, observations):
        return [self.memories[agent_idx].get_observation() for agent_idx, obs in enumerate(observations)] 

    def clear(self):
        self.memories = None


class ObsMemoryWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_radius):
        super().__init__(env)
        self.obs_radius = obs_radius

        size = self.obs_radius * 2 + 1
        self.observation_space: gymnasium.spaces.Sequence = gymnasium.spaces.Sequence( # 定义输出的形状？
            gymnasium.spaces.Dict(        
                obstacles=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
                agents=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
                xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                direction = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
            )
        )

        self.mobsm = MultipleObsMemory()

    def observation(self, observations):
        self.mobsm.update(observations)
        observations_with_memory = self.mobsm.get_observations_with_memory(observations)
        return observations_with_memory

    def reset(self,seed=None, **kwargs):
        self.mobsm.clear()
        obs , infos = self.env.reset(seed=None, **kwargs)
        return self.observation(obs) , infos
