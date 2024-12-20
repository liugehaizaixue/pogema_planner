import matplotlib.pyplot as plt
import pandas as pd
import json
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他字体，如 'Microsoft YaHei'（Windows系统） 或 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


with open('./result/ASwitcher/ASwitcher.jsonl', 'r') as f:
    as_data = [json.loads(line) for line in f]

with open('./result/HSwitcher/HSwitcher_3.jsonl', 'r') as f:
    hs_data = [json.loads(line) for line in f]

with open('./result/MSwitcher/MSwitcher_9.jsonl', 'r') as f:
    ms_data = [json.loads(line) for line in f]


total_data =[]

for i in range(len(as_data)):
    data = {
        'num_agents': as_data[i]['num_agents'],
        'replan_usage': 0,
        'epom_usage': 1,
        'as_usage': as_data[i]['total']['AVG_learning'] / (as_data[i]['total']['AVG_planning'] + as_data[i]['total']['AVG_learning']),
        'hs_usage': hs_data[i]['total']['AVG_learning'] / (hs_data[i]['total']['AVG_planning'] + hs_data[i]['total']['AVG_learning']),
        'ms_usage': ms_data[i]['total']['AVG_learning'] / (ms_data[i]['total']['AVG_planning'] + ms_data[i]['total']['AVG_learning'])
    }
    total_data.append(data)



df = pd.DataFrame(total_data)

markers = ['o', 's', '^', 'D','v','X']  # 圆圈、正方形、向上的三角形
linestyles = ['-', '--', ':', '-.','-.','-.']  # 前1实线，后三虚线

plt.plot(df['num_agents'], df['replan_usage'], label='Replan' ,marker="o", linestyle="-")
plt.plot(df['num_agents'], df['epom_usage'], label='EPOM', marker="s", linestyle="--")
plt.plot(df['num_agents'], df['as_usage'], label='ASwitcher' , marker="^", linestyle=":")
plt.plot(df['num_agents'], df['hs_usage'], label='HSwitcher', marker="D", linestyle="-.")
plt.plot(df['num_agents'], df['ms_usage'], label='MSwitcher', marker="v", linestyle="-.")
plt.xlabel('智能体数量')
plt.ylabel('EPOM 使用率')
# plt.title(title)
plt.legend()
plt.grid(True)
plt.savefig('EPOM使用率比较图.png')
plt.show()
