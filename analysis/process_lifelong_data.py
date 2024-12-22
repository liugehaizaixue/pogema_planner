import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他字体，如 'Microsoft YaHei'（Windows系统） 或 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



with open('total_time_cost.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

with open('./result/baseline/epom-对比-在理想视野下训练的带旋转动作-在vpo-mapf环境下测试_lifelong.jsonl', 'r') as f:
    baseline_data = [json.loads(line) for line in f]

with open('./result/LSwitcher/LSwitcher_lifelong.jsonl', 'r') as f:
    ls_data = [json.loads(line) for line in f]


for i in range(len(data)):
    data[i]['v-epom_seconds'] = data[i]['epom_seconds']
    data[i]['epom_seconds'] = baseline_data[i]['total']['cost_time']*10
    data[i]['ls_seconds'] = ls_data[i]['total']['cost_time']*10

with open('total_time_cost.json', 'w') as f:
    json.dump(data, f)