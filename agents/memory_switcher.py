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

class MSwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['MSwitcher'] = 'MSwitcher'

    memory_length: int = 6
    num_agents_to_switch: int = 1


class MemorySwitcher(SwitcherBase):
    def __init__(self,cfg:MSwitcherConfig):
        self.memories = []
        self.cfg = cfg

    def update_memories(self, observations):
        if not self.memories:
            self.memories = [ [] for obs in observations]
        for idx, obs in enumerate(observations):
            self.memories[idx].append(1 if np.sum(obs['agents'] == 1) > self.cfg.num_agents_to_switch else 0) 
        for mem in self.memories:
            if len(mem) >self.cfg.memory_length:
                mem.pop(0)

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        self.update_memories(observations)
        return [ 1 if any(mem) else 0 for mem in self.memories]


    def after_reset(self):
        super().after_reset()
        self.memories = []


def example_memory_switcher(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64,
                               main_dir='./', animate=False, on_target="finish"):
    from agents.epom import EpomConfig
    algo = MemorySwitcher(MSwitcherConfig(learning=EpomConfig(path_to_weights=str(main_dir / Path('weights/epom')))))
    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate, on_target)


if __name__ == '__main__':
    print(example_memory_switcher(main_dir='../'))
