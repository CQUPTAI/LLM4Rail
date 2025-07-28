import pandas as pd
from service.user_info import UserInfo
from service.modules import Tool

ticketquery_desc = '''车票查询：本接口用于从数据库中查询符合用户要求的火车票。接口输入格式：{"起始站":<起始高铁站>, "终点站":<终点高铁站>, "发车日期":<发车日期>, "到站日期":<到站日期>, "最早发车时刻":<最早发车时刻>, "最晚发车时刻":<最晚发车时刻>, "最早到站时刻":<最早到站时刻>, "最晚到站时刻":<最晚到站时刻>}，其中：时刻的格式都应该形如"08:15"、日期的格式都应该形如"2025-06-07"、必须包含所有参数、参数值可以填None来表示不作要求、但起始站和终点站的值不可为None、出发日期缺失时请填写当前日期、起始站缺失时请填写用户所在地的车站'''

class TicketQuery(Tool):
    def __init__(self, name="车票查询", description=ticketquery_desc):
        super().__init__(name, description)
        self.tickets = pd.read_csv('dataset/ticket.csv')
        self.tickets.drop('软卧/动卧/一等卧', axis=1, inplace=True)
        self.tickets = self.tickets.groupby(['起始站', '终点站']).apply(lambda X: X)
        self.tickets['出发日期'] = pd.to_datetime(self.tickets['出发日期'], format="mixed")
        self.tickets['到达日期'] = pd.to_datetime(self.tickets['到达日期'], format='mixed')
        self.tickets['出发时间'] = pd.to_datetime(self.tickets['出发时间'])
        self.tickets['到达时间'] = pd.to_datetime(self.tickets['到达时间'])

    def __call__(self, parameter: dict, user_info: UserInfo, history: list) -> dict:
        l_info = []
        src = parameter['起始站']
        des = parameter['终点站']
        s_date = pd.Timestamp(parameter['发车日期'])
        e_date = pd.Timestamp(parameter['到站日期'])
        l_dept_time = pd.Timestamp(parameter['最早发车时刻'])
        r_dept_time = pd.Timestamp(parameter['最晚发车时刻'])
        l_arrv_time = pd.Timestamp(parameter['最早到站时刻'])
        r_arrv_time = pd.Timestamp(parameter['最晚到站时刻'])
        results = self.tickets

        try:
            results = results.loc[src]
            results = results.loc[des]
            results = results[results['出发日期'] == s_date]
            results = results[results['到达日期'] == e_date]
            results = results[results['出发时间'] >= l_dept_time]
            results = results[results['出发时间'] <= r_dept_time]
            results = results[results['到达时间'] >= l_arrv_time]
            results = results[results['到达时间'] <= r_arrv_time]
        except Exception:
            l_info.append(None)

        if len(l_info) == 0:
            for _, result in results.iterrows():
                info = {}
                for idx in result.index.to_list():
                    if str(result[idx]) != 'nan':
                        info[idx] = result[idx]
                info['出发日期'] = info['出发日期'].strftime("%Y-%m-%d")
                info['到达日期'] = info['到达日期'].strftime("%Y-%m-%d")
                info['出发时间'] = info['出发时间'].strftime("%H:%M:%S")
                info['到达时间'] = info['到达时间'].strftime("%H:%M:%S")
                l_info.append(info)
            if len(l_info) > 5:
                l_info = l_info[:5]

        return l_info