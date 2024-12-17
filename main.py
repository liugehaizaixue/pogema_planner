import os
from prettytable import PrettyTable
import time
from agents.epom import example_epom
from agents.replan import example_replan
from agents.assistant_switcher import example_assistant_switcher
from agents.memory_switcher import example_memory_switcher
from agents.density_switcher import example_density_switcher
from agents.heuristic_switcher import example_heuristic_switcher
# from agents.centralized import example_cbs

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# def example(map_name='hit', num_agents=64, seed=0, animate=True):
def example(map_name='sc1-AcrosstheCape', num_agents=64, seed=0, animate=True):
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"

    run_examples_funcs = [
        example_epom,
        example_replan,
        example_memory_switcher,
        example_density_switcher,
        example_heuristic_switcher,
        example_assistant_switcher,
        # example_cbs,
    ]

    score_table = PrettyTable()
    # score_table.field_names = ["Algorithm", "ISR", "CSR", "Episode Length"]
    score_table.field_names = ["Algorithm", "ISR", "CSR", "Episode Length", "ConflictNums"]

    for run_example_func in run_examples_funcs:
        start_time = time.time()
        result = run_example_func(map_name=map_name, num_agents=num_agents, seed=seed, max_episode_steps=512, animate=animate)
        end_time = time.time()
        print("Time: ", end_time - start_time)
        if result:
            # score_table.add_row([result['algorithm'], result['ISR'], result['CSR'], result['ep_length'] ])
            score_table.add_row([result['algorithm'], result['ISR'], result['CSR'], result['ep_length'], result['conflict_nums']])

            print(score_table.get_string(start=len(score_table._rows) - 1, end=len(score_table._rows)))

    print(score_table)


if __name__ == '__main__':
    start_time = time.time()
    example()
    end_time = time.time()
    print("Time: ", end_time - start_time)
