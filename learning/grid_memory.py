import gymnasium
import numpy as np
from gymnasium.spaces import Box
from pogema import GridConfig
from pogema.grid import Grid
from copy import deepcopy

class GridMemory:
    def __init__(self, start_r=32 , memory_type="default"):
        self.memory_type = memory_type
        if memory_type == "default":
            self.obstacles_memory = np.zeros(shape=(start_r * 2 + 1, start_r * 2 + 1))  # memory.shape[0] 起始就是32
        elif memory_type == "plus":
            self.obstacles_memory = np.full(shape=(start_r * 2 + 1, start_r * 2 + 1), fill_value=-1)
        elif memory_type == "max":
            self.obstacles_memory = np.full(shape=(start_r * 2 + 1, start_r * 2 + 1), fill_value=-1)
            self.agents_memory = np.full(shape=(start_r * 2 + 1, start_r * 2 + 1), fill_value=-1)  #-1未知，0已知无，1 ... 4 当前观察机器人-过去四个记忆内其他机器人分布，即更新记忆时，将大于1的值+1，大于4的值归0

    @staticmethod
    def try_to_insert(x, y, source, target):
        r = source.shape[0] // 2
        try:
            mask = (source >= 0)  # 创建一个布尔掩码，表示source中的元素是否为0或1
            target_slice = target[x - r:x + r + 1, y - r:y + r + 1]
            target_slice[mask] = source[mask]
            # target[x - r:x + r + 1, y - r:y + r + 1] = source
            return True
        except (ValueError, IndexError) as e:
            print(f"An error occurred: {e}")
            return False

    def increase_obstacles_memory(self):
        """ memory矩阵 扩充一倍 """
        m = self.obstacles_memory
        r = self.obstacles_memory.shape[0]
        if self.memory_type == "default":
            self.obstacles_memory = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
        elif self.memory_type == "plus":
            self.obstacles_memory = np.full(shape=(r * 2 + 1, r * 2 + 1), fill_value=-1)
        elif self.memory_type == "max":
            self.obstacles_memory = np.full(shape=(r * 2 + 1, r * 2 + 1), fill_value=-1)

        assert self.try_to_insert(r, r, m, self.obstacles_memory)

    def increase_agents_memory(self):
        """ agents_memory矩阵 扩充一倍 """
        m = self.agents_memory
        r = self.agents_memory.shape[0]
        self.agents_memory = np.full(shape=(r * 2 + 1, r * 2 + 1), fill_value=-1)
        assert self.try_to_insert(r, r, m, self.agents_memory)

    def update_obstacles(self, x, y, obstacles):
        while True:
            r = self.obstacles_memory.shape[0] // 2
            if self.try_to_insert(r + x, r + y, obstacles, self.obstacles_memory):  # 此处r+x,r+y 实际上是将obs中当前位置的坐标系，转换为memory中的坐标系，如0，0对应memory中心的32，32
                break
            self.increase_obstacles_memory()

    def update_agents(self, x, y, other_agents):
        r = other_agents.shape[0] // 2
        other_agents[r, r] = 0 #先将观察中的 中心元素置为0，即不考虑自己 ，最后返回obs时再设为1
        self.agents_memory[self.agents_memory >= 4] = -1 #再将记忆中大于4的记忆丢弃
        self.agents_memory[self.agents_memory >= 0] += 1 #并将过去所有智能体记忆+1
        # 然后再插入新的观察        
        while True:
            r = self.agents_memory.shape[0] // 2
            if self.try_to_insert(r + x, r + y, other_agents, self.agents_memory):  # 此处r+x,r+y 实际上是将obs中当前位置的坐标系，转换为memory中的坐标系，如0，0对应memory中心的32，32
                break
            self.increase_agents_memory()

    def get_obstacles_observation(self, x, y, obs_radius):
        while True:
            r = self.obstacles_memory.shape[0] // 2
            tx, ty = x + r, y + r
            size = self.obstacles_memory.shape[0]
            if 0 <= tx - obs_radius and tx + obs_radius + 1 <= size:
                if 0 <= ty - obs_radius and ty + obs_radius + 1 <= size:
                    return self.obstacles_memory[tx - obs_radius:tx + obs_radius + 1, ty - obs_radius:ty + obs_radius + 1]

            self.increase_obstacles_memory()

    def get_agents_observation(self, x, y, obs_radius):
        while True:
            r = self.agents_memory.shape[0] // 2
            tx, ty = x + r, y + r
            size = self.agents_memory.shape[0]
            if 0 <= tx - obs_radius and tx + obs_radius + 1 <= size:
                if 0 <= ty - obs_radius and ty + obs_radius + 1 <= size:
                    return self.agents_memory[tx - obs_radius:tx + obs_radius + 1, ty - obs_radius:ty + obs_radius + 1]

            self.increase_agents_memory()

    def render(self):
        m = self.obstacles_memory.astype(int).tolist()
        gc = GridConfig(map=m)
        g = Grid(add_artificial_border=False, grid_config=gc)
        r = self.obstacles_memory.shape[0] // 2
        g.positions_xy = [[r, r]]
        g.finishes_xy = []
        g.render()


class MultipleGridMemory:
    def __init__(self, memory_type="default"):
        self.memories = None
        self.memory_type = memory_type

    def update(self, observations): #更改memory中的obstacles
        if self.memories is None or len(self.memories) != len(observations):
            self.memories = [GridMemory(memory_type=self.memory_type) for _ in range(len(observations))]
        for agent_idx, obs in enumerate(observations):
            self.memories[agent_idx].update_obstacles(*obs['xy'], obs['obstacles'])
            if self.memory_type == "max":
                self.memories[agent_idx].update_agents(*obs['xy'], obs['agents'])

    def get_obstacles_observations(self, xy_list, obs_radius):
        return [self.memories[idx].get_obstacles_observation(x, y, obs_radius) for idx, (x, y) in enumerate(xy_list)]  # 从记忆中获取obs

    def get_agents_observations(self, xy_list, obs_radius):
        return [self.memories[idx].get_agents_observation(x, y, obs_radius) for idx, (x, y) in enumerate(xy_list)]  # 从记忆中获取obs
    
    def modify_observation(self, observations, obs_radius):
        all_xy = [observations[idx]['xy'] for idx in range(len(observations))]
        for obs, gm_obstacles in zip(observations, self.get_obstacles_observations(all_xy, obs_radius)):  # 观察的obs_radius(env_cfg.grid_config.obs_radius)可能是5，而从记忆中拿去的obs_radius（env_cfg.grid_memory_obs_radius）是7
            obs['obstacles'] = gm_obstacles

        if self.memory_type == "max":
            for obs, gm_agents in zip(observations, self.get_agents_observations(all_xy, obs_radius)):  # 观察的obs_radius(env_cfg.grid_config.obs_radius)可能是5，而从记忆中拿去的obs_radius（env_cfg.grid_memory_obs_radius）是7
                obs['agents'] = deepcopy(gm_agents)
                # 记忆不考虑自身的位置，因此再此处额外加入自己位置，即中心设为1
                r = gm_agents.shape[0] // 2
                obs['agents'][r, r] = 1 
        else:
            r = obs_radius # env_cfg.grid_memory_obs_radius = 7
            rr = observations[0]['agents'].shape[0] // 2 # env_cfg.grid_config.obs_radius = 5
            for agent_idx, obs in enumerate(observations):  # 修改agent矩阵大小，但agent没有记忆中的位置
                if self.memory_type == "default":
                    if rr <= r:
                        agents = np.zeros(shape=(r * 2 + 1, r * 2 + 1)) #不可见区域全为0
                        mask = (obs['agents'] >= 0)  # 创建一个布尔掩码，表示obs['agents']中的元素是否为0或1
                        agents_slice = agents[r - rr:r + rr + 1, r - rr: r + rr + 1]
                        agents_slice[mask] = obs['agents'][mask]
                        obs['agents'] = agents
                    else:
                        obs['agents'] = obs['agents'][rr - r:rr + r + 1, rr - r: rr + r + 1]
                        agents = np.zeros_like(obs['agents'])
                        mask = (obs['agents'] >= 0)
                        agents[mask] = obs['agents'][mask]
                        obs['agents'] = agents
                elif self.memory_type == "plus":
                    if rr <= r:
                        agents = np.full(shape = (r * 2 + 1, r * 2 + 1), fill_value=-1)  # 创建-1矩阵 ，原本不可见区域全为0，而现在存在不可见区域设为-1
                        agents[r - rr:r + rr + 1, r - rr: r + rr + 1] = obs['agents']
                        obs['agents'] = agents
                    else:
                        obs['agents'] = obs['agents'][rr - r:rr + r + 1, rr - r: rr + r + 1]


    def clear(self):
        self.memories = None


class GridMemoryWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_radius, memory_type="default"):
        super().__init__(env)
        self.obs_radius = obs_radius
        size = self.obs_radius * 2 + 1

        if memory_type == "default":
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(size, size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(size, size)),
                xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                direction = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
            )
        elif memory_type == "plus":
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
                agents=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
                xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                direction = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
            )
        elif memory_type == "max":
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
                agents=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
                xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
                direction = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
            )

        self.mgm = MultipleGridMemory(memory_type=memory_type)

    def observation(self, observations):
        self.mgm.update(observations)
        self.mgm.modify_observation(observations, self.obs_radius)
        return observations

    def reset(self,seed=None, **kwargs):
        self.mgm.clear()
        obs , infos = self.env.reset(seed=None, **kwargs)
        return self.observation(obs) , infos
