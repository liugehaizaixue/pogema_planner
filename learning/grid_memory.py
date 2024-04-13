import gymnasium
import numpy as np
from gymnasium.spaces import Box
from pogema import GridConfig
from pogema.grid import Grid


class GridMemory:
    def __init__(self, start_r=32):
        # self.memory = np.zeros(shape=(start_r * 2 + 1, start_r * 2 + 1))  # memory.shape[0] 起始就是32
        self.memory = np.full(shape=(start_r * 2 + 1, start_r * 2 + 1), fill_value=-1)

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

    def increase_memory(self):
        """ memory矩阵 扩充一倍 """
        m = self.memory
        r = self.memory.shape[0]
        # self.memory = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
        self.memory = np.full(shape=(r * 2 + 1, r * 2 + 1), fill_value=-1)
        assert self.try_to_insert(r, r, m, self.memory)

    def update(self, x, y, obstacles):
        while True:
            r = self.memory.shape[0] // 2
            if self.try_to_insert(r + x, r + y, obstacles, self.memory):  # 此处r+x,r+y 实际上是将obs中当前位置的坐标系，转换为memory中的坐标系，如0，0对应memory中心的32，32
                break
            self.increase_memory()

    def get_observation(self, x, y, obs_radius):
        while True:
            r = self.memory.shape[0] // 2
            tx, ty = x + r, y + r
            size = self.memory.shape[0]
            if 0 <= tx - obs_radius and tx + obs_radius + 1 <= size:
                if 0 <= ty - obs_radius and ty + obs_radius + 1 <= size:
                    return self.memory[tx - obs_radius:tx + obs_radius + 1, ty - obs_radius:ty + obs_radius + 1]

            self.increase_memory()

    def render(self):
        m = self.memory.astype(int).tolist()
        gc = GridConfig(map=m)
        g = Grid(add_artificial_border=False, grid_config=gc)
        r = self.memory.shape[0] // 2
        g.positions_xy = [[r, r]]
        g.finishes_xy = []
        g.render()


class MultipleGridMemory:
    def __init__(self):
        self.memories = None

    def update(self, observations): #更改memory中的obstacles
        if self.memories is None or len(self.memories) != len(observations):
            self.memories = [GridMemory() for _ in range(len(observations))]
        for agent_idx, obs in enumerate(observations):
            self.memories[agent_idx].update(*obs['xy'], obs['obstacles'])

    def get_observations(self, xy_list, obs_radius):
        return [self.memories[idx].get_observation(x, y, obs_radius) for idx, (x, y) in enumerate(xy_list)]  # 从记忆中获取obs

    def modify_observation(self, observations, obs_radius):
        all_xy = [observations[idx]['xy'] for idx in range(len(observations))]
        for obs, gm_obs in zip(observations, self.get_observations(all_xy, obs_radius)):  # 观察的obs_radius(env_cfg.grid_config.obs_radius)可能是5，而从记忆中拿去的obs_radius（env_cfg.grid_memory_obs_radius）是7
            obs['obstacles'] = gm_obs

        r = obs_radius # env_cfg.grid_memory_obs_radius = 7
        rr = observations[0]['agents'].shape[0] // 2 # env_cfg.grid_config.obs_radius = 5
        for agent_idx, obs in enumerate(observations):  # 修改agent矩阵大小，但agent没有记忆中的位置

            if rr <= r:
                # agents = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
                agents = np.full(shape = (r * 2 + 1, r * 2 + 1), fill_value=-1)  # 创建-1矩阵 ，原本不可见区域全为0，而现在存在不可见区域设为-1
                agents[r - rr:r + rr + 1, r - rr: r + rr + 1] = obs['agents']
                obs['agents'] = agents
            else:
                obs['agents'] = obs['agents'][rr - r:rr + r + 1, rr - r: rr + r + 1]

    def clear(self):
        self.memories = None


class GridMemoryWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_radius):
        super().__init__(env)
        self.obs_radius = obs_radius

        size = self.obs_radius * 2 + 1
        self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
            obstacles=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
            agents=gymnasium.spaces.Box(-1.0, 1.0, shape=(size, size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            direction = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
        )

        self.mgm = MultipleGridMemory()

    def observation(self, observations):
        self.mgm.update(observations)
        self.mgm.modify_observation(observations, self.obs_radius)
        return observations

    def reset(self,seed=None, **kwargs):
        self.mgm.clear()
        obs , infos = self.env.reset(seed=None, **kwargs)
        return self.observation(obs) , infos
