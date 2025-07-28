import pandas as pd
import random
from collections import defaultdict

df_ticket = pd.read_csv("dataset/ticket.csv")

d_stations = ['上海虹桥', '拉萨', '南宁', '北京西', '杭州西']
a_stations = []
random.seed(610)
all_stations = df_ticket['终点站'].unique().tolist()
for station in d_stations:
    able = df_ticket[df_ticket['起始站']==station].loc[:, '终点站'].unique().tolist()
    unable = list(set(all_stations)-set(able))
    random.shuffle(able)
    random.shuffle(unable)
    a_stations.append(able[:3]+unable[:2])
all_arrivals = []
for i in range(len(a_stations)):
    all_arrivals.extend(a_stations[i])
dates = ['2025-5-11', '2025-5-13', '2025-5-15', '2025-5-17', '2025-5-19', '2025-5-21', '2025-5-22', '2025-5-24', '2025-5-26',
         '今天', '明天', '后天',
         '下周三', '下周四', '下周五','下周六', '下下周一', '下下周二', '下下周日',
         '下周', '下下周', '下周末', '下下周末']
times = ['上午', '下午', '晚上',
         '上午8点之前', '上午9点之前', '上午9点之后', '上午10点之后',
         '下午2点之前', '下午4点之前', '下午2点之后', '下午3点之后',
         '晚上9点之前', '晚上11点之前', '晚上7点之后', '晚上6点之后',
         '上午9点左右', '上午11点左右',
         '下午1点左右', '下午4点左右',
         '晚上7点左右', '晚上10点左右']

queries = []

template = '查询从{departure_station}到{destination_station}，在{arrival_date}{arrival_time}到达的车次'
for date in dates:
    for time in times:
        for i, departure in enumerate(d_stations):
            for arrival in a_stations[i]:
                queries.append(f'A1,{date} {time} {departure} {arrival},'+template.format(arrival_date=date, arrival_time=time,
                                                                                          departure_station=departure, destination_station=arrival))

template = '查询在{arrival_date}{arrival_time}到达{destination_station}的车次'
for date in dates:
    for time in times:
        for arrival in all_arrivals:
            queries.append(f'A2,{date} {time} {arrival},'+template.format(arrival_date=date, arrival_time=time,
                                                                          destination_station=arrival))

template = '查询{departure_date}{departure_time}出发，从{departure_station}到{destination_station}的车次'
for date in dates:
    for time in times:
        for i, departure in enumerate(d_stations):
            for arrival in a_stations[i]:
                queries.append(f'B1,{date} {time} {departure} {arrival},'+template.format(departure_date=date, departure_time=time,
                                                                                          departure_station=departure, destination_station=arrival))

template = '查询{departure_date}{departure_time}出发，到{destination_station}的车次'
for date in dates:
    for time in times:
        for arrival in all_arrivals:
            queries.append(f'B2,{date} {time} {arrival},'+template.format(departure_date=date, departure_time=time,
                                                                          destination_station=arrival))

random.seed(1024)
random.shuffle(queries)
queries = queries[:975]

template = '查询从{departure_station}到{destination_station}的车次'
for i, departure in enumerate(d_stations):
    for arrival in a_stations[i]:
        queries.append(f'C,{departure} {arrival},'+template.format(departure_station=departure, destination_station=arrival))

print(f'总样本数：{len(queries)}')

count = defaultdict(int)
for query in queries:
    if 'A1' in query:
        count['A1'] += 1
    elif 'A2' in query:
        count['A2'] += 1
    elif 'B1' in query:
        count['B1'] += 1
    elif 'B2' in query:
        count['B2'] += 1
    elif 'C' in query:
        count['C'] += 1
print(f'样本类型分布：{dict(count)}')

with open("dataset/ticket_all.txt", encoding='utf-8',mode='w') as f:
    f.write('\n'.join(queries))