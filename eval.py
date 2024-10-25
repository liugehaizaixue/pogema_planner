import os
from pathlib import Path
from prettytable import PrettyTable
from pomapf_env.custom_maps import MAPS_REGISTRY , _test_regexp
import re
import time
import signal
from agents.epom import EPOM , EpomConfig
from agents.replan import RePlan , RePlanConfig
from agents.utils_eval import eval_algorithm
from multiprocessing import Pool, cpu_count

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

def evaluate_algorithm_on_map(args):
    algo_name, map_name, num_agents, seed = args
    algo = get_algo_by_name(algo_name)
    print(f"current seed: {seed} , current map: {map_name}")
    result = eval_algorithm(algo=algo , map_name=map_name, num_agents=num_agents, seed=seed, max_episode_steps=512, animate=False)
    return result

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    # os.environ['OMP_NUM_THREADS'] = "1"
    # os.environ['MKL_NUM_THREADS'] = "1"
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    algo_list = ["Replan" , "EPOM"]

    score_table = PrettyTable()
    score_table.field_names = ["Algorithm", "Num of Agents","Avg ISR", "Avg CSR", "Avg Episode Length", "Avg ConflictNums"]

    test_maps = get_test_maps()

    num_processes = min(cpu_count(), 10)  # 使用的进程数为CPU核心数和地图数量中较小的那个
    print(f"num_processes: {num_processes}")
    pool = Pool(processes=num_processes , initializer=init_worker)
    for algo_name in algo_list:
        print(f"current algo_name: {algo_name}")
        for num_agents in [64 , 128 , 192 , 256, 320, 384]:
            print(f"current num_agents: {num_agents}")
            _step = 0
            _ISR = 0
            _CSR = 0
            _ep_length = 0
            _conflict_nums = 0

            # 构造参数列表
            args_list = [(algo_name, map_name, num_agents, seed) for map_name in test_maps for seed in range(10)]

            # 多进程地评估算法在不同地图上的性能
            results = pool.map_async(evaluate_algorithm_on_map, args_list)
            try:
                # 等待所有任务完成或手动中断
                while not results.ready() and results._number_left != 0:
                    time.sleep(1)

                all_results = results.get()
                for result in all_results:
                    _step += 1
                    _ISR += result['ISR']
                    _CSR += result['CSR']
                    _ep_length += result['ep_length']
                    _conflict_nums += result['conflict_nums']
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                os.kill(os.getpid(), signal.SIGINT)

            score_table.add_row([algo_name, num_agents , _ISR/_step, _CSR/_step, _ep_length/_step , _conflict_nums/_step])
            new_row = score_table.get_string(start=len(score_table._rows) - 1, end=len(score_table._rows))
            print(new_row)
            write_into_file(score_table, algo_name, current_time)

    pool.close()
    pool.join()

    print(score_table)

if __name__ == '__main__':
    main()
