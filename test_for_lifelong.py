import os
from prettytable import PrettyTable
import time
# from agents.epom import example_epom
from agents.replan import example_replan

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def example(map_name='sc1-AcrosstheCape', num_agents=64, seed=0, animate=True, max_episode_steps=32, on_target="restart"):
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"

    run_examples_funcs = [
        # example_epom,
        example_replan,
    ]

    score_table = PrettyTable()
    score_table.field_names = ["Algorithm", "AVG THROUGHPUT", "ConflictNums"]

    for run_example_func in run_examples_funcs:
        result = run_example_func(map_name=map_name, num_agents=num_agents, seed=seed, max_episode_steps=max_episode_steps, animate=animate, on_target=on_target)

        if result:
            score_table.add_row([result['algorithm'], result['avg_throughput'], result['conflict_nums']])

            print(score_table.get_string(start=len(score_table._rows) - 1, end=len(score_table._rows)))

    print(score_table)


if __name__ == '__main__':
    start_time = time.time()
    example()
    end_time = time.time()
    print("Time: ", end_time - start_time)
