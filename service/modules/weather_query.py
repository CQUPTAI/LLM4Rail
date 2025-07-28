import Levenshtein
import numpy as np
import os, requests, json
from service.modules import Tool
from service.user_info import UserInfo

city_file_path = r'dataset\citycode.json'
weatherquery_desc = '''天气查询：本接口用于查询天气。接口输入格式：{"城市名":<用户想要查询的城市名>, "日期":<用户希望查询的日期>}，其中<日期>格式应该形如："2025-06-07"（月和日均为两位数）'''

class WeatherQuery(Tool):
    def __init__(self, name="天气查询", description=weatherquery_desc):
        super().__init__(name, description)

        self.url = "https://restapi.amap.com/v3/weather/weatherInfo?parameters"
        self.key = os.environ.get('amapKey', '')

        with open(city_file_path, 'r', encoding="utf-8") as f:
            self.urlCity = json.load(f)
        self.city_list = self._getCity()
        self.city_names = [city['name'] for city in self.city_list]

    def _getCity(self):
        city = []
        if "data" in self.urlCity:
            cityByLetter = self.urlCity["data"]["cityByLetter"]
            for k, v in cityByLetter.items():
                city.extend(v)
        return city

    def normalized_similarity(self, city1:str, city2: str):
        distance = Levenshtein.distance(city1, city2)
        max_len = max(len(city1), len(city2))
        if max_len == 0:
            return 1.0
        return 1 - (distance / max_len)

    def fuzzy_search(self, query_city: str):
        scores = []
        for city_name in self.city_names:
            score = self.normalized_similarity(query_city, city_name)
            scores.append(score)
        sorted_idx = np.array(scores).argsort().tolist()[::-1]
        return [self.city_names[idx] for idx in sorted_idx[:5]]

    def __call__(self, parameter: dict, user_info: UserInfo, history: list) -> dict:
        info = []

        adcode = list(filter(lambda d: d["name"] == parameter['城市名'], self.city_list))
        if len(adcode) == 0:
            info.append(f"查询失败，{parameter['城市名']}不是一个合法的城市名！你可能想查询的是：{'、'.join(self.fuzzy_search(parameter['城市名']))}。")
        else:
            adcode = adcode[0]['adcode']
            params = {'key': self.key,
                      'city': adcode,
                      'extensions': 'all'}
            response = requests.get(self.url, params=params)
            casts = response.json()['forecasts'][0]['casts']
            for cast in casts:
                if cast['date'] == parameter['日期']:
                    info.append(cast)
            if len(info) == 0:
                info.append(f'查询失败，只支持查询未来3天的天气！')

        return info