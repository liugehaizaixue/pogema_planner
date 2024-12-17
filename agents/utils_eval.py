import pathlib
import torch
from pogema.animation import AnimationConfig, AnimationMonitor
from pomapf_env.env import make_pomapf
from pomapf_env.pomapf_config import POMAPFConfig


def eval_algorithm(algo, map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64, animate=False, on_target="finish"):
    gc = POMAPFConfig(map_name=map_name, max_episode_steps=max_episode_steps, seed=seed, num_agents=num_agents, on_target=on_target)
    env = make_pomapf(grid_config=gc)
    algo_name = type(algo).__name__
    if animate:
        anim_dir = str(pathlib.Path('renders') / algo_name)
        env = AnimationMonitor(env, AnimationConfig(egocentric_idx=0 , directory=anim_dir))
    obs, infos = env.reset()
    algo.after_reset()
    results_holder = ResultsHolder()

    dones = [False for _ in range(len(obs))]
    rew = [0 for _ in range(len(obs))]
    actions = algo.act(obs, rew, dones, infos)

    with torch.no_grad():
        while True:
            obs, rew, dones, truncated, infos = env.step(actions)
            results_holder.after_step(infos)
            algo.after_step(dones)

            if all(dones) or all(truncated):
                break

            actions = algo.act(obs, rew, dones, infos)
    try:
        usage = algo.get_usage()
    except:
        usage = {
            'learning': 0,
            'planning': 0
        }
    results = results_holder.get_final()
    results['learning'] = usage['learning']
    results['planning'] = usage['planning']
    results['algorithm'] = algo_name
    return results


class ResultsHolder:
    def __init__(self):
        self.results = dict()

    def after_step(self, infos):
        if 'metrics' in infos[0]:
            self.results.update(**infos[0]['metrics'])

    def get_final(self):
        return self.results
