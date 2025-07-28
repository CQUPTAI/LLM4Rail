import argparse
from datetime import date
from collections import defaultdict
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--saveDir', '-s', required=True, action='store')
args = parser.parse_args()
result_file = f'{args.saveDir}/results.json'

with open(result_file, encoding='utf-8') as f:
    lines = f.readlines()
samples = [eval(line.strip()) for line in lines]

city_mapping = {}
with open('evaluation/weather/analysis/city_mapping.txt', encoding='utf-8') as f:
    reverse_mapping = eval(f.read())
for key, val in reverse_mapping.items():
    for district in val:
        city_mapping[district] = key

# WARN: YOU MUST UPDATE THIS FILE WITH THE CURRENT DATE!!!
date_mapping = {}
with open('evaluation/weather/analysis/date_mapping(7.20).txt', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    label, possibles = line.strip().split('   ')
    date_mapping[label] = eval(possibles)

tokenizer = AutoTokenizer.from_pretrained('evaluation/tokenizer/')

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

cur_date = date.today().strftime("%Y-%m-%d")
def find_city_and_dates(q_type, label, user_info):
    loc_city = find_city(user_info)
    target_city, target_date = None, None
    if q_type == 'A':
        label_date, label_city = label.split(' ')
        target_city = label_city
        target_date = date_mapping[label_date]
    elif q_type == 'B':
        label_date = label
        target_city = loc_city
        target_date = date_mapping[label_date]
    elif q_type == 'C':
        label_city = label
        target_city = label_city
        target_date = date_mapping[cur_date]
    if q_type == 'D':
        label_date, label_city = label.split(' ')
        target_city = city_mapping[label_city]
        target_date = date_mapping[label_date]
    elif q_type == 'E':
        label_date = label
        target_city = loc_city
        target_date = date_mapping[label_date]
    elif q_type == 'F':
        label_city = label
        target_city = city_mapping[label_city]
        target_date = date_mapping[cur_date]
    return target_city, target_date

correct, total = 0, len(samples)
success_turns = defaultdict(int)
trial_stat = defaultdict(int)
stats_err = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int),
             'D': defaultdict(int), 'E': defaultdict(int), 'F': defaultdict(int)}
stats_err_type = defaultdict(int)
stats_suc_type = defaultdict(int)
total_turns, total_tokens = 0, 0
for sample in samples:
    user_info = sample['user_info']
    q_type, label = sample['query_type'], sample['label']
    traj = sample['trajectory']
    true_city, true_dates_tuple = find_city_and_dates(q_type, label, user_info)

    success, success_turn = False, None
    stat_dates = defaultdict(int)
    for n, cycle in enumerate(traj[:-1]):
        action_input = cycle['action_input']
        city, date = action_input['城市名'], action_input['日期']
        if city==true_city:
            for i, true_dates in enumerate(true_dates_tuple):
                if date in true_dates:
                    stat_dates[i] = 1
                    if not success:
                        success_turn = n+1
                        success = True
                    break
        total_tokens += len(tokenizer.tokenize(cycle['completion']))
    cnt_match = 0
    for i in range(len(true_dates_tuple)):
        cnt_match += stat_dates[i]
    if cnt_match == len(true_dates_tuple):
        correct += 1
        stats_suc_type[q_type] += 1
        success_turns[success_turn] += 1
    else:
        stats_err_type[q_type] += 1
        if len(traj) > 1:   # Only the last cycle
            cycle = traj[-2]
            action_input = cycle['action_input']
            city, date = action_input['城市名'], action_input['日期']
            if city != true_city:
                stats_err[q_type]['city'] += 1
            date_match = False
            for i, true_dates in enumerate(true_dates_tuple):
                if date in true_dates:
                    date_match = True
                    break
            if not date_match:
                stats_err[q_type]['date'] += 1
    total_turns += len(traj)
    if len(traj) > 1:
        trial_stat[sample['trial_count']] += 1
    else:
        trial_stat['Failed'] += 1


# Success Rate
print(f'API调用的成功率：{correct/total*100:.2f}% \n')

# Number of Trials
print(f'尝试次数分布：\n{dict(trial_stat)}\n')

# Number of QTAO Iterations to Success
success_turns = {key:val/correct for key, val in success_turns.items()}
print('成功调用API在第几轮：\n', dict(success_turns), '\n')

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
print(stats_errtype)
print()

# Average Number of QTAO Iterations
print(f'平均的TAO循环数：{total_turns / len(samples):.4f}\n')

# Average Number of Tokens per QTAO Iteration
print(f'每个TAO循环的平均token数：{total_tokens / total_turns:.4f}\n')