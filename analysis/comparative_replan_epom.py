import pandas as pd
import json

def read_file(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
    return data

files = ['./result/replan.jsonl','./result/replan.jsonl'] 
replan_data = read_file(files[0])
epom_data = read_file(files[1])
# 将 JSONL 字符串分行处理，每行是一个 JSON 对象
records_replan = [json.loads(line) for line in replan_data.strip().split('\n') if line.strip()]
records_epom = [json.loads(line) for line in replan_data.strip().split('\n') if line.strip()]

total_records = [('replan', records_replan) , ('epom', records_epom)]
# 初始化一个空的 DataFrame
df = pd.DataFrame(columns=[
    'num_of_agents', 'algorithm', 'sc1_ISR', 'sc1_CSR', 'sc1_avg_length',
    'street_ISR', 'street_CSR', 'street_avg_length',
    'wc3_ISR', 'wc3_CSR', 'wc3_avg_length',
    'mazes_ISR', 'mazes_CSR', 'mazes_avg_length',
    'random_ISR', 'random_CSR', 'random_avg_length'
])

# 遍历记录并填充 DataFrame
for algorithm_name, records in total_records:
    for record in records:
        row = {
            'num_of_agents': record['num_agents'],
            'algorithm': algorithm_name,  # 替换为实际算法名称
            'sc1_ISR': record['sc1']['ISR'] / record['sc1']['counts'],
            'sc1_CSR': record['sc1']['CSR'] / record['sc1']['counts'],
            'sc1_avg_length': record['sc1']['ep_length'] / record['sc1']['counts'],
            'street_ISR': record['street']['ISR'] / record['street']['counts'],
            'street_CSR': record['street']['CSR'] / record['street']['counts'],
            'street_avg_length': record['street']['ep_length'] / record['street']['counts'],
            'wc3_ISR': record['wc3']['ISR'] / record['wc3']['counts'],
            'wc3_CSR': record['wc3']['CSR'] / record['wc3']['counts'],
            'wc3_avg_length': record['wc3']['ep_length'] / record['wc3']['counts'],
            'mazes_ISR': record['mazes']['ISR'] / record['mazes']['counts'],
            'mazes_CSR': record['mazes']['CSR'] / record['mazes']['counts'],
            'mazes_avg_length': record['mazes']['ep_length'] / record['mazes']['counts'],
            'random_ISR': record['random']['ISR'] / record['random']['counts'],
            'random_CSR': record['random']['CSR'] / record['random']['counts'],
            'random_avg_length': record['random']['ep_length'] / record['random']['counts'],
            'avg_ISR': record['total']['AVG_ISR'],
            'avg_CSR': record['total']['AVG_CSR'],
            'avg_length': record['total']['AVG_ep_length']
        }
        new_row_df = pd.DataFrame([row])  # 将字典转换为 DataFrame
        df = pd.concat([df, new_row_df], ignore_index=True)

df = df.sort_values(by=['num_of_agents', 'algorithm'])
# 输出 DataFrame，可以进一步将其导出为 Excel 文件等
print(df)
df.to_excel('output.xlsx', index=False)
