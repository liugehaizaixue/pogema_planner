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
import json

SEEDS = 1
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

def match_maps(map_name):
    # 根据前几个字符找出map所属于的类型
    patterns = ['wc3', 'sc1', 'street', 'mazes', 'random']
    for pattern in patterns:
        if map_name[:3] in pattern :
            return pattern
    print("No matching pattern found for map_name:", map_name)
    return None

def write_into_file(metrics, algo_name, current_time, task_type="lifelong"):
    path = './eval_result/lifelong/'
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = "eval_result/lifelong/"+ algo_name+"_"+task_type+"_"+current_time+".jsonl"
    with open(file_name,"a") as f:
        json.dump(metrics, f)
        f.write("\n")
        f.close()

def get_total_metrics(metrics):
    total_data = {
        "throughput": 0,
        "conflict_nums": 0,
        "counts": 0
    }
    for map_type, metric in metrics.items():
        if map_type == 'num_agents':
            continue
        for  key , value in metric.items():
            total_data[key] += value
    
    total_data['AVG_THROUGHPUT'] = total_data['throughput'] / total_data['counts']
    total_data['AVG_conflict_nums'] = total_data['conflict_nums'] / total_data['counts']
    return total_data
    

def evaluate_algorithm_on_map(args):
    algo_name, map_name, num_agents, seed = args
    algo = get_algo_by_name(algo_name)
    print(f"current seed: {seed} , current map: {map_name}")
    result = eval_algorithm(algo=algo , map_name=map_name, num_agents=num_agents, seed=seed, max_episode_steps=512, animate=False, on_target="restart")
    result['map_name'] = map_name
    return result

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    # os.environ['OMP_NUM_THREADS'] = "1"
    # os.environ['MKL_NUM_THREADS'] = "1"
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    algo_list = ["Replan" , "EPOM"]

    score_table = PrettyTable()
    score_table.field_names = ["Algorithm", "Num of Agents", "AVG THROUGHPUT", "AVG ConflictNums"]

    test_maps = get_test_maps()

    num_processes = min(cpu_count(), 10)  # 使用的进程数为CPU核心数和地图数量中较小的那个
    print(f"num_processes: {num_processes}")
    pool = Pool(processes=num_processes , initializer=init_worker)
    for algo_name in algo_list:
        print(f"current algo_name: {algo_name}")
        for num_agents in [32, 64 , 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]:
            print(f"current num_agents: {num_agents}")
            metrics = {
                'num_agents': num_agents,
            }

            # 构造参数列表
            args_list = [(algo_name, map_name, num_agents, seed) for map_name in test_maps for seed in range(SEEDS)]

            # 多进程地评估算法在不同地图上的性能
            results = pool.map_async(evaluate_algorithm_on_map, args_list)
            try:
                # 等待所有任务完成或手动中断
                while not results.ready() and results._number_left != 0:
                    time.sleep(1)

                all_results = results.get()
                for result in all_results:
                    map_type = match_maps(result['map_name'])
                    if map_type not in metrics:
                        metrics[map_type] = {'throughput': 0, 'conflict_nums': 0, 'counts': 0}
                    metrics[map_type]['throughput'] += result['avg_throughput']
                    metrics[map_type]['conflict_nums'] += result['conflict_nums']
                    metrics[map_type]['counts'] += 1
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                os.kill(os.getpid(), signal.SIGINT)

            total_metrics = get_total_metrics(metrics)
            metrics['total'] = total_metrics
            score_table.add_row([algo_name, num_agents , total_metrics['AVG_THROUGHPUT'], total_metrics['AVG_conflict_nums']])
            new_row = score_table.get_string(start=len(score_table._rows) - 1, end=len(score_table._rows))
            print(new_row)
            write_into_file(metrics, algo_name, current_time, task_type="lifelong")

    pool.close()
    pool.join()

    print(score_table)

if __name__ == '__main__':
    main()
