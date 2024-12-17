from pathlib import Path

from agents.utils_agents import run_algorithm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra

from agents.replan import RePlanConfig
from agents.utils_switching import SwitcherBaseConfig, SwitcherBase
import numpy as np

class DSwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['DSwitcher'] = 'DSwitcher'
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=False, no_path_random=True,
                                          use_best_move=True, fix_nones=True)
    density_to_switch: float = 0.5


class DensitySwitcher(SwitcherBase):

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        return [ 1 if self.get_density(obs)>self.cfg.density_to_switch else 0 for obs in observations]


    def get_density(self, observation):
        agents = np.sum( observation['agents'] == 1) # 至少为1，因为有自己
        obstacles = np.sum( observation['obstacles'] == 1) # 至少为0
        all_size = np.sum( observation['agents'] != -1) # 至少为2 ，不可能为1
        # free_size = np.sum( observation['obstacles'] == 0) # 最少为1，即面前只有一个障碍物，此时agents一定是1
        # if free_size == 1: 
        #     density = 0
        # else:
        #     density = (agents-1) / (free_size - 1)  # 考虑 障碍物密度 
        density = (obstacles + agents-1) / (all_size - 1)  # 不考虑 障碍物密度 
        return density



def example_density_switcher(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64,
                               main_dir='./', animate=False, on_target="finish"):
    from agents.epom import EpomConfig
    algo = DensitySwitcher(DSwitcherConfig(learning=EpomConfig(path_to_weights=str(main_dir / Path('weights/epom')))))
    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate, on_target)


if __name__ == '__main__':
    print(example_density_switcher(main_dir='../'))
