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
files = ['./result/epom/epom-best.jsonl','./result/epom/epom-default-best.jsonl',
         './result/epom/epom-no-direction-best.jsonl', './result/epom/epom_网格记忆消融_best.jsonl']  # 添加你的文件名
algorithm_names = ['EPOM' ,'EPOM (filled with 0)','EPOM (no direction)','EPOM (no Grid Memory)']
markers = ['o', 's', '^', 'D','v','X']  # 圆圈、正方形、向上的三角形
linestyles = ['-', '-.', '-.', '-.','-.','-.']  # 前1实线，后三虚线
# 解析每个文件并抽取需要的数据
all_data = []

for file in files:
    df = read_jsonl(file)
    all_data.append(df[['num_agents', 'total.AVG_CSR', 'total.AVG_ISR', 'total.AVG_ep_length', 'total.AVG_conflict_nums']])

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
    plt.ylabel(ylabel, fontsize=14)
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'EPOM消融-{ylabel}.png')
    plt.show()

plot_metric('total.AVG_CSR', '整体成功率', 'Average CSR by Algorithm')
plot_metric('total.AVG_ISR', '独立成功率', 'Average ISR by Algorithm')
plot_metric('total.AVG_ep_length', '平均回合长度', 'Average Episode Length by Algorithm')
# plot_metric('total.AVG_conflict_nums', 'Average Conflict Nums', 'Average Conflict Nums by Algorithm')