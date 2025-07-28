import random
from datetime import date, timedelta

weather_query_templates = ['查一下{date}{city}的天气',
                           '查一下{date}的天气',
                           '查一下{city}的天气']

with open('dataset/city_names', encoding='utf-8') as f:
    city_names = f.readlines()
city_names = [city_name.strip() for city_name in city_names]

now_date = date.today()
dates = [(now_date+timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0, 6)]

weather_query_clear = []
for date in dates:
    for city_name in city_names:
        weather_query_clear.append(f'A,{date} {city_name},'+weather_query_templates[0].format(date=date, city=city_name))
for city_name in city_names:
    weather_query_clear.append(f'C,{city_name},'+weather_query_templates[2].format(city=city_name))
random.seed(1805)
random.shuffle(weather_query_clear)
weather_query_clear = weather_query_clear[:294]
for date in dates:
    weather_query_clear.append(f'B,{date},'+weather_query_templates[1].format(date=date))
count = {'A':0, 'B':0, 'C':0}
for sample in weather_query_clear[:300]:
    query_type, label, query = tuple(sample.split(','))
    count[query_type] += 1

weather_query_clear = '\n'.join(weather_query_clear)
with open(f'dataset/weather_clear({now_date.strftime("%Y-%m-%d")}).txt', encoding='utf-8', mode='w') as f:
    f.write(weather_query_clear)

amb_dates = ["今天","明天","后天","下周一","下周二","未来2天",  "昨天","大后天","下周三",  "下周","最近","这两天","未来几天"]
city_district_pairs  = {"北京": ["东城区", "西城区", "朝阳区", "海淀区", "丰台区"],
               "上海": ["黄浦区", "徐汇区", "浦东新区", "长宁区", "静安区"],
               "广州": ["天河区", "越秀区", "荔湾区", "海珠区", "白云区"],
               "深圳": ["福田区", "罗湖区", "南山区", "宝安区", "龙岗区"],
               "重庆": ["渝中区", "江北区", "南岸区", "九龙坡区", "渝北区"],
               "成都": ["锦江区", "青羊区", "金牛区", "武侯区", "成华区"],
               "武汉": ["江岸区", "江汉区", "硚口区", "汉阳区", "武昌区"],
               "杭州": ["上城区", "拱墅区", "西湖区", "滨江区", "萧山区"],
               "西安": ["碑林区", "雁塔区", "未央区", "莲湖区", "新城区"],
               "拉萨": ["城关区", "堆龙德庆区", "达孜区", "林周县", "当雄县"]}
amb_cities = []
for val in city_district_pairs.values():
    amb_cities.extend(val)

weather_query_amb = []
for date in amb_dates:
    for city_name in amb_cities:
        weather_query_amb.append(f'D,{date} {city_name},'+weather_query_templates[0].format(date=date, city=city_name))
for city_name in amb_cities:
    weather_query_amb.append(f'F,{city_name},'+weather_query_templates[2].format(city=city_name))
random.seed(1805)
random.shuffle(weather_query_amb)
weather_query_amb = weather_query_amb[:487]
for date in amb_dates:
    weather_query_amb.append(f'E,{date},'+weather_query_templates[1].format(date=date))

weather_query_amb = '\n'.join(weather_query_amb)
with open(f'dataset/weather_amb({now_date.strftime("%Y-%m-%d")}).txt', encoding='utf-8', mode='w') as f:
    f.write(weather_query_amb)
