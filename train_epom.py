import faulthandler
faulthandler.enable()

import json
from argparse import Namespace
from sys import argv
import yaml
from sample_factory.train import make_runner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.utils.utils import log
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
import wandb

from learning.epom_config import Experiment
from learning.register_env import register_custom_components
from learning.register_training_utils import register_custom_model, register_msg_handlers

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_sf_config(exp: Experiment):
    custom_argv = [f'--env={exp.env}']
    parser, partial_cfg = parse_sf_args(argv=custom_argv, evaluation=False)
    parser.set_defaults(**exp.dict())
    final_cfg = parse_full_cfg(parser, argv=custom_argv)
    return final_cfg


def run(config=None):

    if config is None:
        import argparse

        parser = argparse.ArgumentParser(description='Process training config.')

        parser.add_argument('--config_path', type=str, action="store", default='train-debug.yaml',
                            help='path to yaml file with single run configuration', required=False)

        parser.add_argument('--raw_config', type=str, action='store',
                            help='raw json config', required=False)

        parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                            help='Run wandb in thread mode. Usefull for some setups.', required=False)

        params = parser.parse_args()
        if params.raw_config:
            params.raw_config = params.raw_config.replace("\'", "\"")
            config = json.loads(params.raw_config)
        else:
            if params.config_path is None:
                raise ValueError("You should specify --config_path or --raw_config argument!")
            with open(params.config_path, "r") as f:
                config = yaml.safe_load(f)
    else:
        params = Namespace(**config)
        params.wandb_thread_mode = False

    exp = Experiment(**config)
    flat_config = Namespace(**exp.dict())
    model_type = exp.environment.model_type
    register_custom_model(model_type)
    env_name = exp.environment.env
    log.debug(f'env_name = {env_name}')
    register_custom_components(env_name)

    log.info(flat_config)

    if exp.train_for_env_steps == 1_000_000:
        exp.use_wandb = False

    if exp.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project='Pogema-Planner', config=exp.dict(), save_code=False, sync_tensorboard=True,
                   anonymous="allow", job_type=exp.environment.env, group='train')

    flat_config, runner = make_runner(create_sf_config(exp))
    register_msg_handlers(flat_config, runner)
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status

def recursive_update(experiment: dict, key, value):
    if key in experiment:
        experiment[key] = value
        return True
    else:
        for k, v in experiment.items():
            if isinstance(v, dict):
                if recursive_update(v, key, value):
                    return True
        return False


def update_dict(target_dict, keys, values):
    for key, value in zip(keys, values):
        if recursive_update(target_dict, key, value):
            print(f'Updated {key} to {value}')
        else:
            raise KeyError(f'Could not find {key} in experiment')


def parse_args_to_items(argv_):
    keys = []
    values = []

    for arg in argv_[1:]:
        key, value = arg.split('=')
        key = key.replace('--', '')

        keys.append(key)
        values.append(value)

    return keys, values

def main():
    experiment = Experiment()
    experiment = create_sf_config(experiment).__dict__
    keys, values = parse_args_to_items(list(argv))

    # check all args and replace them in experiment recursively
    update_dict(experiment, keys, values)
    run(config=experiment)

if __name__ == '__main__':
    main()