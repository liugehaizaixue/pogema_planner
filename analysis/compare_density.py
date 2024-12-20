import pandas as pd
import matplotlib.pyplot as plt
import json

# plt.style.use('ggplot')
# 定义一个函数来读取和解析 JSONL 文件
def read_jsonl(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.json_normalize(data)

# 文件名列表，假设你的文件放在当前目录下
files = ['./result/replan/replan_density0.0.jsonl','./result/replan/replan_density0.1.jsonl','./result/replan/replan_density0.2.jsonl','./result/replan/replan_density0.3.jsonl','./result/test-epom-density0.3.jsonl']  # 添加你的文件名
algorithm_names = ['Replan(density=0)','Replan(density=0.1)', 'Replan(density=0.2)', 'Replan(density=0.3)','EPOM(density=0.3)']
markers = ['o', 's', '^', 'D', 'X']  # 圆圈、正方形、向上的三角形
linestyles = ['-', '-.', '-.', '-.', '-.']  # 前1实线，后三虚线
# 解析每个文件并抽取需要的数据
all_data = []

for file in files:
    df = read_jsonl(file)
    all_data.append(df[['num_agents', 'total.AVG_CSR', 'total.AVG_ISR', 'total.AVG_ep_length']])

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
    plt.xlabel('Number of Agents')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_metric('total.AVG_CSR', 'Average CSR', 'Average CSR by Algorithm')
plot_metric('total.AVG_ISR', 'Average ISR', 'Average ISR by Algorithm')
plot_metric('total.AVG_ep_length', 'Average Episode Length', 'Average Episode Length by Algorithm')