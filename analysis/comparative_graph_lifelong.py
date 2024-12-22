import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他字体，如 'Microsoft YaHei'（Windows系统） 或 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# plt.style.use('ggplot')
# 定义一个函数来读取和解析 JSONL 文件
def read_jsonl(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.json_normalize(data)

# 文件名列表，假设你的文件放在当前目录下
files = ['./result/replan/replan_lifelong.jsonl','./result/epom/epom-lifelong.jsonl', './result/ASwitcher/ASwitcher_lifelong.jsonl', './result/HSwitcher/HSwitcher_lifelong.jsonl', './result/MSwitcher/MSwitcher_lifelong.jsonl', './result/LSwitcher/LSwitcher_lifelong.jsonl','./result/baseline/epom-对比-在理想视野下训练的带旋转动作-在vpo-mapf环境下测试_lifelong.jsonl']  # 添加你的文件名
algorithm_names = ['Replan','V-EPOM', 'ASwitcher', 'HSwitcher', 'MSwitcher','LSwitcher','EPOM']
markers = ['o', 's', '^', 'D','v','X','+']  
linestyles = ['-', '--', ':', '-.','-.','-.','-.'] 
# 解析每个文件并抽取需要的数据
all_data = []

for file in files:
    df = read_jsonl(file)
    all_data.append(df[['num_agents', 'total.AVG_THROUGHPUT']])

# 合并所有数据，方便比较
combined_data = pd.concat(all_data, keys=algorithm_names)

# 重置索引，以便绘图
combined_data.reset_index(level=0, inplace=True)
combined_data.rename(columns={'level_0': 'Algorithm'}, inplace=True)


def plot_metric(metric, ylabel, title):
    plt.figure(figsize=(10, 6))
    for name, marker , linestyle in zip(algorithm_names, markers, linestyles):
        data = combined_data[combined_data['Algorithm'] == name]
        plt.plot(data['num_agents'], data[metric], label=name, marker=marker, linestyle=linestyle)
        # plt.scatter(data['num_agents'], data[metric], s=50, marker=marker)  # 强调数据点
    plt.xlabel('智能体数量', fontsize=14)
    plt.ylabel('平均吞吐量', fontsize=14)
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'终身MAPF对比图.png')
    plt.show()

plot_metric('total.AVG_THROUGHPUT', 'AVG_THROUGHPUT', 'Average THROUGHPUT by Algorithm')
