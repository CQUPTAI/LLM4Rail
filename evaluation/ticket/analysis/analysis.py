import argparse
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--saveDir', '-s', required=True, action='store')
args = parser.parse_args()
result_file = f'{args.saveDir}/results.json'

with open(result_file, encoding='utf-8') as f:
    lines = f.readlines()
samples = [eval(line.strip()) for line in lines]

date_mapping = {}
with open('evaluation/ticket/analysis/date_mapping', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    label, possibles = line.strip().split('   ')
    date_mapping[label] = eval(possibles)

time_mapping = {}
with open('evaluation/ticket/analysis/time_mapping', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    label, tuple = line.strip().split('   ')
    time_mapping[label] = eval(tuple)

city_stations = None
with open('evaluation/ticket/analysis/city_stations.json', encoding='utf-8') as f:
    city_stations = eval(f.read().strip())
for city, stations in city_stations.items():
    for i in range(len(stations)):
        if stations[i][-1] == '站':
            stations[i] = stations[i][:-1]

def find_city(user_info):
    location = user_info['所在地']
    if '自治州' in location:
        return location.split()[1]
    elif '市' in location.split()[0]:
        return location.split()[0][:-1]
    else:
        if location.split()[1][-1] == '市':
            return location.split()[1][:-1]
        else:
            return location.split()[1]

tokenizer = AutoTokenizer.from_pretrained('evaluation/tokenizer/')

def judge(loc_city, cycle, q_type, label):
    action_input = cycle['action_input']
    dept_station, dest_station = action_input['起始站'], action_input['终点站']
    s_date, e_date = action_input['发车日期'], action_input['到站日期']
    l_dept_time, r_dept_time = pd.Timestamp(action_input['最早发车时刻']), pd.Timestamp(action_input['最晚发车时刻'])
    l_arrv_time, r_arrv_time = pd.Timestamp(action_input['最早到站时刻']), pd.Timestamp(action_input['最晚到站时刻'])
    if pd.isna(l_dept_time):
        l_dept_time = pd.Timestamp("00:00")
    if pd.isna(l_arrv_time):
        l_arrv_time = pd.Timestamp('00:00')
    if pd.isna(r_dept_time):
        r_dept_time = pd.Timestamp("23:59")
    if pd.isna(r_arrv_time):
        r_arrv_time = pd.Timestamp("23:59")

    is_correct, err_types = True, set()
    if q_type == 'A1':
        true_e_date, true_e_time, true_dept_station, true_dest_station = label.split(' ')
        true_l_arrv_time = pd.Timestamp(time_mapping[true_e_time][0])
        true_r_arrv_time = pd.Timestamp(time_mapping[true_e_time][1])
        if dept_station != true_dept_station:
            is_correct = False
            err_types.add('station')
        if dest_station != true_dest_station:
            is_correct = False
            err_types.add('station')
        if e_date not in date_mapping[true_e_date]:
            is_correct = False
            err_types.add('date')
        if l_arrv_time > true_l_arrv_time or r_arrv_time < true_r_arrv_time:
            is_correct = False
            err_types.add('time')
    elif q_type == 'A2':
        true_e_date, true_e_time, true_dest_station = label.split(' ')
        true_l_arrv_time = pd.Timestamp(time_mapping[true_e_time][0])
        true_r_arrv_time = pd.Timestamp(time_mapping[true_e_time][1])
        if dept_station not in city_stations[loc_city]:
            is_correct = False
            err_types.add('station')
        if dest_station != true_dest_station:
            is_correct = False
            err_types.add('station')
        if e_date not in date_mapping[true_e_date]:
            is_correct = False
            err_types.add('date')
        if l_arrv_time > true_l_arrv_time or r_arrv_time < true_r_arrv_time:
            is_correct = False
            err_types.add('time')
    elif q_type == 'B1':
        true_s_date, true_s_time, true_dept_station, true_dest_station = label.split(' ')
        true_l_dept_time = pd.Timestamp(time_mapping[true_s_time][0])
        true_r_dept_time = pd.Timestamp(time_mapping[true_s_time][1])
        if dept_station != true_dept_station:
            is_correct = False
            err_types.add('station')
        if dest_station != true_dest_station:
            is_correct = False
            err_types.add('station')
        if s_date not in date_mapping[true_s_date]:
            is_correct = False
            err_types.add('date')
        if l_dept_time > true_l_dept_time or r_dept_time < true_r_dept_time:
            is_correct = False
            err_types.add('time')
    elif q_type == 'B2':
        true_s_date, true_s_time, true_dest_station = label.split(' ')
        true_l_dept_time = pd.Timestamp(time_mapping[true_s_time][0])
        true_r_dept_time = pd.Timestamp(time_mapping[true_s_time][1])
        if dept_station not in city_stations[loc_city]:
            is_correct = False
            err_types.add('station')
        if dest_station != true_dest_station:
            is_correct = False
            err_types.add('station')
        if s_date not in date_mapping[true_s_date]:
            is_correct = False
            err_types.add('date')
        if l_dept_time > true_l_dept_time or r_dept_time < true_r_dept_time:
            is_correct = False
            err_types.add('time')
    elif q_type == 'C':
        true_dept_station, true_dest_station = label.split(' ')
        if true_dept_station != dept_station:
            is_correct = False
            err_types.add('station')
        if true_dest_station != dest_station:
            is_correct = False
            err_types.add('station')
    return is_correct, list(err_types)

correct, total = 0, len(samples)
success_turn = defaultdict(int)
trial_stat = defaultdict(int)
stats_err = {'A1': defaultdict(int), 'A2': defaultdict(int),
             'B1': defaultdict(int), 'B2': defaultdict(int),
             'C': defaultdict(int)}
stats_err_type = defaultdict(int)
stats_suc_type = defaultdict(int)
total_turns, total_tokens = 0, 0

for sample in samples:
    loc_city = find_city(sample['user_info'])
    q_type, label = sample['query_type'], sample['label']
    traj = sample['trajectory']
    flag = False
    for n, cycle in enumerate(traj[:-1]):
        if sample['trial_count'] == 1:
            is_correct, err_types = judge(loc_city, cycle, q_type, label)
            if is_correct:
                flag = True
                success_turn[n+1] += 1
                break
        total_tokens += len(tokenizer.tokenize(cycle['completion']))
    if flag:
        correct += 1
        stats_suc_type[q_type] += 1
    else:
        stats_err_type[q_type] += 1
        if len(traj) > 1:
            cycle = traj[-2]
            is_correct, err_types = judge(loc_city, cycle, q_type, label)
            for err_type in err_types:
                stats_err[q_type][err_type] += 1
    total_turns += len(traj)
    if len(traj) > 1:
        trial_stat[sample['trial_count']] += 1
    else:
        trial_stat['Failed'] += 1

# Success Rate
print(f'API调用的成功率：{correct/total*100:.2f}% \n')

# Number of Trials
trial_stat = {key:val/len(samples) for key,val in trial_stat.items()}
print(f'尝试次数分布：\n{dict(trial_stat)}\n')

# Number of QTAO Iterations to Success
success_turn = {key:val/correct for key, val in success_turn.items()}
print('成功调用API在第几轮：\n', dict(success_turn), '\n')

# Success Rate for Different Types of Queries
print('对于每种查询类型，成功调用API的概率：')
for key in stats_err.keys():
    print(f'{key}: {stats_suc_type[key]/(stats_err_type[key]+stats_suc_type[key])*100:.2f}%({stats_err_type[key]+stats_suc_type[key]})', end=', ')
print('\n')

# Failure Reasons
print('对于每种类型的查询，调用API失败的原因：')
stats_errtype = defaultdict(int)
for key in stats_err.keys():
    print(f'{key}: ', dict(stats_err[key]))
    for type in stats_err[key].keys():
        stats_errtype[type] += stats_err[key][type]
stats_errtype = dict(stats_errtype)
total_err = 0
for val in stats_errtype.values():
    total_err += val
for key in stats_errtype.keys():
    stats_errtype[key] /= total_err
print(stats_errtype)
print()

# Average Number of QTAO Iterations
print(f'平均的TAO循环数：{total_turns / len(samples):.4f}\n')

# Average Number of Tokens per QTAO Iteration
print(f'每个TAO循环的平均token数：{total_tokens / total_turns:.4f}\n')