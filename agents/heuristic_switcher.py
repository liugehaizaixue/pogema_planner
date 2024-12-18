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

class HSwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['HSwitcher'] = 'HSwitcher'
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=False, no_path_random=True,
                                          use_best_move=True, fix_nones=True)

    num_agents_to_switch: int = 3 #3  #最多可以看到（4+1）个智能体


class HeuristicSwitcher(SwitcherBase):

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        return [ np.sum(obs['agents'] == 1) > self.cfg.num_agents_to_switch for obs in observations]


def example_heuristic_switcher(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64,
                               main_dir='./', animate=False, on_target="finish"):
    from agents.epom import EpomConfig
    algo = HeuristicSwitcher(HSwitcherConfig(learning=EpomConfig(path_to_weights=str(main_dir / Path('weights/epom')))))
    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate, on_target)


if __name__ == '__main__':
    print(example_heuristic_switcher(main_dir='../'))
