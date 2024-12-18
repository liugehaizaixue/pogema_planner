import matplotlib.pyplot as plt
import pandas as pd
import json
# Define the data as seen in the user's table
data = {
    'step':[32,64,96,128,160,192,224,256,288,320,352,384],
    "replan": ["0:16:37", "0:33:11", "0:47:22", "1:04:24", "1:24:03", "1:36:20", "1:51:33",
               "2:07:28", "2:18:17", "2:40:22", "2:57:58", "3:07:39"],
    "epom": ["0:03:02", "0:05:26", "0:07:51", "0:10:08", "0:12:28", "0:14:50", "0:17:09",
             "0:19:34", "0:21:57", "0:24:38", "0:27:01", "0:29:31"]
}

with open('cost_data.json', 'w') as json_file:
    json.dump(data, json_file)

# 从 JSON 文件读取数据
with open('cost_data.json', 'r') as json_file:
    loaded_data = json.load(json_file)

# 转换成 DataFrame
df = pd.DataFrame(loaded_data)

# Create a DataFrame
df = pd.DataFrame(data)

# Define a function to convert time string HH:MM:SS to seconds
def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

# Convert time columns to seconds
df['replan_seconds'] = df['replan'].apply(time_to_seconds)
df['epom_seconds'] = df['epom'].apply(time_to_seconds)

df['replan_step_per_sec'] = df['step']*512*10 / df['replan_seconds']
df['epom_step_per_sec'] = df['step']*512*10 / df['epom_seconds']

df = df[['step','replan_step_per_sec', 'epom_step_per_sec']]

print(df)


# # Plotting
plt.figure(figsize=(10, 6))
width = 3  # width of the bars

# Positions for the bars
positions = df['step'].values
replan_positions = positions + width/2
epom_positions = positions - width/2

plt.bar(replan_positions, df['replan_step_per_sec'], width=width, label='Replan Seconds')
plt.bar(epom_positions, df['epom_step_per_sec'], width=width, label='Epom Seconds')

plt.xlabel('Step')
plt.ylabel('Seconds')
plt.title('Comparison of Replan and Epom Seconds by Step')
plt.xticks(positions)  # Ensure ticks match steps
plt.legend()

plt.show()