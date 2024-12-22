import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他字体，如 'Microsoft YaHei'（Windows系统） 或 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



with open('total_time_cost.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)


df['replan_step_per_sec'] = df['step']*512*10 / df['replan_seconds']
df['v-epom_step_per_sec'] = df['step']*512*10 / df['v-epom_seconds']
df['AS_step_per_sec'] = df['step']*512*10 / df['as_seconds']
df['HS_step_per_sec'] = df['step']*512*10 / df['hs_seconds']
df['MS_step_per_sec'] = df['step']*512*10 / df['ms_seconds']
df['LS_step_per_sec'] = df['step']*512*10 / df['ls_seconds']
df['epom_step_per_sec'] = df['step']*512*10 / df['epom_seconds']

print(df)


# # Plotting
plt.figure(figsize=(10, 6))
width = 3  # width of the bars

# Positions for the bars
positions = df['step'].values
offset = width   # 设置错位的偏移量

# 每个数据集的偏移位置
replan_positions = positions + offset
v_epom_positions = positions - offset
as_positions = positions + 2 * offset
hs_positions = positions - 2 * offset
ms_positions = positions
ls_positions = positions - 3 * offset
epom_positions = positions+3*offset


plt.bar(replan_positions, df['replan_step_per_sec'], width=width, label='Replan')
plt.bar(v_epom_positions, df['v-epom_step_per_sec'], width=width, label='V-EPOM')
plt.bar(ms_positions, df['MS_step_per_sec'], width=width, label='MSwitcher')
plt.bar(as_positions, df['AS_step_per_sec'], width=width, label='ASwitcher')
plt.bar(hs_positions, df['HS_step_per_sec'], width=width, label='HSwitcher')
plt.bar(ls_positions, df['LS_step_per_sec'], width=width, label='LSwitcher')
plt.bar(epom_positions, df['epom_step_per_sec'], width=width, label='EPOM')


plt.xlabel('智能体数量', fontsize=14)
plt.ylabel('每秒执行次数', fontsize=14)
# plt.title('Comparison of Replan and Epom Seconds by Step')
plt.xticks(positions)  # Ensure ticks match steps
plt.legend()
plt.savefig('时间消耗图.png')
plt.show()