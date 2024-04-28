try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra
from learning.grid_memory import MultipleGridMemory
from learning.obs_memory import MultipleObsMemory
from learning.epom_config import Environment
from copy import deepcopy
from agents.utils_agents import AlgoBase, run_algorithm
from planning.replan_algo import RePlanBase, FixLoopsWrapper, NoPathSoRandomOrStayWrapper, FixNonesWrapper


class RePlanConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['A-with-direction', 'A-star'] = 'A-with-direction'
    num_process: int = 5
    fix_loops: bool = True
    no_path_random: bool = True
    fix_nones: bool = True
    add_none_if_loop: bool = False
    use_best_move: bool = True
    stay_if_loop_prob: float = 0.5
    max_planning_steps: int = 10000
    device: str = 'cpu'


class RePlan:
    def __init__(self, cfg: RePlanConfig):
        self.cfg = cfg
        self.agent = None
        self.fix_loops = cfg.fix_loops
        self.fix_nones = cfg.fix_nones
        self.stay_if_loop_prob = cfg.stay_if_loop_prob
        self.no_path_random = cfg.no_path_random
        self.use_best_move = cfg.use_best_move
        self.add_none_if_loop = cfg.add_none_if_loop
        if cfg.name == 'A-with-direction':
            self.algo_name = 'A-with-direction'
        else:
            self.algo_name = 'A-star'
        self.env = None
        self.mgm = MultipleGridMemory(memory_type="default")
        # self.mobsm = MultipleObsMemory()

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        test_observations = deepcopy(observations)
        self.mgm.update(test_observations)
        gm_radius = 7
        self.mgm.modify_observation(test_observations, obs_radius=gm_radius)
        return self.agent.act(observations, skip_agents)

    def after_step(self, dones):
        if all(dones):
            self.agent = None

    def after_reset(self, ):
        self.agent = RePlanBase(use_best_move=self.use_best_move, max_steps=self.cfg.max_planning_steps,
                                algo_name=self.algo_name, seed=self.cfg.seed)

        if self.fix_loops:
            self.agent = FixLoopsWrapper(self.agent, stay_if_loop_prob=self.stay_if_loop_prob,
                                         add_none_if_loop=self.add_none_if_loop)
        if self.no_path_random:
            self.agent = NoPathSoRandomOrStayWrapper(self.agent)
        elif self.fix_nones:
            self.agent = FixNonesWrapper(self.agent)


def example_replan(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64, animate=False):
    algo = RePlan(RePlanConfig(max_planning_steps=10000))
    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate)


if __name__ == '__main__':
    print(example_replan())
