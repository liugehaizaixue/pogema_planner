try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pogema import GridConfig


class POMAPFConfig(GridConfig):
    integration: Literal['SampleFactory'] = 'SampleFactory'
    collision_system: Literal['block_both','priority'] = 'priority'
    observation_type: Literal['POMAPF', 'MAPF'] = 'POMAPF'
    on_target: Literal['finish'] = 'finish'
    auto_reset: Literal[False] = False
    num_agents: int = 8
    obs_radius: int = 5
    max_episode_steps: int = 1024
    map_name: str = '(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)'
