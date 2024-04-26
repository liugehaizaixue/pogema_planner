import os
from pathlib import Path
from prettytable import PrettyTable
from pomapf_env.custom_maps import MAPS_REGISTRY , _test_regexp
import re
import time
from agents.epom import EPOM , EpomConfig
from agents.replan import RePlan , RePlanConfig
from agents.utils_eval import eval_algorithm

def get_algo_by_name(algo_name):
    if algo_name == "Replan":
        algo = RePlan(RePlanConfig(max_planning_steps=10000))
    elif algo_name == "EPOM":
        algo = EPOM(EpomConfig(path_to_weights=str('./' / Path('weights/epom'))))
    return algo

def get_test_maps():
    test_maps = []
    pattern = _test_regexp
    for map_name in MAPS_REGISTRY:
        if not re.match(pattern, map_name):
            test_maps.append(map_name)
    return test_maps

def write_into_file(x, algo_name , current_time):
    path = './eval_result/'
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = "eval_result/"+ algo_name+"_"+current_time
    with open(file_name,"w") as f:
        f.write(str(x))
        f.close()


def main():
    # os.environ['OMP_NUM_THREADS'] = "1"
    # os.environ['MKL_NUM_THREADS'] = "1"
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    algo_list = ["Replan" , "EPOM"]

    score_table = PrettyTable()
    score_table.field_names = ["Algorithm", "Num of Agents","Avg ISR", "Avg CSR", "Avg Episode Length"]

    test_maps = get_test_maps()
    for algo_name in algo_list:
        print(f"current algo_name: {algo_name}")
        algo = get_algo_by_name(algo_name)
        for num_agents in [64 , 128 , 192 , 256, 320, 384]:
            print(f"current num_agents: {num_agents}")
            _step = 0
            _ISR = 0
            _CSR = 0
            _ep_length = 0
            for seed in range(0,10):
                print(f"current seed: {seed}")
                for map_name in test_maps:
                    print(f"current map: {map_name}")
                    result = eval_algorithm(algo=algo , map_name=map_name, num_agents=num_agents, seed=seed, max_episode_steps=512, animate=False)
                    algo.after_reset()
                    _step = _step + 1
                    _ISR = _ISR + result['ISR']
                    _CSR = _CSR + result['CSR']
                    _ep_length = _ep_length + result['ep_length']

            score_table.add_row([algo_name, num_agents , _ISR/_step, _CSR/_step, _ep_length/_step])
            new_row = score_table.get_string(start=len(score_table._rows) - 1, end=len(score_table._rows))
            print(new_row)
            write_into_file(score_table, algo_name, current_time)
    print(score_table)



if __name__ == '__main__':
    main()