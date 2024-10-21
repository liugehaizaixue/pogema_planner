import os
from prettytable import PrettyTable
import time
# from agents.epom import example_epom
from agents.replan import example_replan

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def example(map_name='sc1-AcrosstheCape', num_agents=64, seed=0, animate=True):
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"

    run_examples_funcs = [
        # example_epom,
        example_replan,
    ]

    score_table = PrettyTable()
    # score_table.field_names = ["Algorithm", "ISR", "CSR", "Episode Length"]
    score_table.field_names = ["Algorithm", "ISR", "CSR", "Episode Length", "ConflictNums"]

    for run_example_func in run_examples_funcs:
        result = run_example_func(map_name=map_name, num_agents=num_agents, seed=seed, max_episode_steps=512, animate=animate)

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
