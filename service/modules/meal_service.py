import numpy as np
import pandas as pd
from openai import OpenAI
import re, json, Levenshtein
from service.modules import Tool
from sklearn.metrics.pairwise import cosine_similarity

recommend_prompt = \
'''
【任务】 基于用户信息和历史对话，推荐用户最有可能想要的{top_k}个餐食
【格式要求】 先进行思考和分析，然后按Python列表格式输出推荐结果：{output_format}！
【用户信息】 {user_info}
【历史对话】 {dialogue}
'''

judge_prompt = \
'''
【任务】 为候选集中的每一个餐食在检索结果中寻找相匹配的餐食，这里“相匹配”指二者是同一样食物
【输出格式】 
先进行思考和分析，然后输出一个Python字典，字典的键是候选集中每一个餐食的名字，对应的值是在检索结果中与之相匹配的某一个食品的名字：{{"<候选餐食1>": "<匹配餐食1>", ..., "<候选餐食n>":"<匹配餐食n>"}}
如果不存在相匹配的餐食，则对应的值为None；
如果有多个相匹配的餐食，则选择匹配程度最高的哪一个。
【候选集】 {raw_candidates}
【检索结果】 {retrieved_items}
'''

encoding_prompt = \
'''
【任务】 
判断给定餐食的特征。
你要判断的特征有：
饮食类型：(1)正餐、(2)小吃、(3)饮料/咖啡、(0)未加限定
菜系：(1)川菜、(2)粤菜、(3)鲁菜、(4)淮扬菜、(5)闽菜、(6)湘菜、(7)西北菜、(8)东北菜、(0)未加限定
中西餐：(1)中餐、(2)西餐、(3)素食、(4)清真、(0)未加限定
辣度：4个1/0，分别表示是否不辣、是否微辣、是否中辣、是否特辣（其中1表示是，0表示不是）
价格：包含两个整数的列表[a, b]或"未加限定"（其中a、b为两个非负整数，且满足a<=b）
北京/天津/河北/山西/内蒙古/辽宁/吉林/黑龙江/上海/江苏/浙江/安徽/福建/江西/山东/河南/湖北/湖南/广东/广西/海南/重庆/四川/贵州/云南/西藏/陕西/甘肃/青海/宁夏/新疆：该餐食对不同省份人群的适合程度，每个值的范围为1~5，1表示最不推荐，5表示最推荐
儿童/青年/成年人/中年人/老年人：该餐食对不同年龄人群的适合程度，每个值的范围为1~5，1表示最不推荐，5表示最推荐
早餐/午餐/晚餐/下午茶/夜宵：该餐食对不同时段的适合程度，每个值的范围为1~5，1表示最不推荐，5表示最推荐
男/女：该餐食对不同性别，每个值的范围为1~5，1表示最不推荐，5表示最推荐
春/夏/秋/冬：该餐食对不同季节的适合程度，每个值的范围为1~5，1表示最不推荐，5表示最推荐
【格式】 
输出要求：以JSON格式输出一个对象数组，每个JSON对象对应一个餐食，对象的键为特征名，值为特征值。注意，饮食类型是1个特征，菜系是1个特征，中西餐是1个特征，辣度有4个特征，省份相关的有31个特征，年龄相关的有5个特征，时段相关的有5个特征，性别相关的有2个特征，季节相关的有4个特征，共52个特征值
输出示例：
```json
[{{
  "饮食类型": <餐食类型>,
  "菜系": <餐食类型>,
  "中西餐": <餐食类型>,
  "不辣": <是否不辣>,
  "微辣": <是否微辣>,
  "中辣": <是否中辣>,
  "特辣": <是否特辣>,
  "价格": <价格>,
  "北京": <适合北京人的程度>,
  "天津": <适合天津人的程度>,
  ...,
  "新疆": <适合新疆人的程度>,
  "儿童": <适合儿童的程度>,
  "青年": <适合青年的程度>,
  "成年人": <适合成年人的程度>,
  "中年人": <适合中年人的程度>,
  "老年人": <适合老年人的程度>,
  "早餐": <适合作为早餐的程度>,
  "午餐": <适合作为午餐的程度>,
  "晚餐": <适合作为晚餐的程度>,
  "下午茶": <适合作为下午茶的程度>,
  "夜宵": <适合作为夜宵的程度>,
  "男": <适合男生的程度>,
  "女": <适合女生的程度>,
  "春": <适合春季的程度>,
  "夏": <适合夏季的程度>,
  "秋": <适合秋季的程度>,
  "冬": <适合冬季的程度>
}}, <描述第二个餐食特征的JSON对象>,...]
```json
【待判断的餐食】
{items}
'''

mealservice_desc = '''餐饮推荐服务：当用户需要推荐餐食时，若通过历史对话你对用户的偏好有充分把握，则可以调用本接口从数据库中匹配最符合用户偏好的菜品；如果你对用户偏好没有足够把握，请不要调用本接口，而是询问用户偏好。接口输入格式：该接口无需输入参数，填None'''

soft_constraints = ["Beijing", "Tianjin", "Hebei", "Shanxi", "Nei Mongol", "Liaoning", "Jilin", "Heilongjiang", "Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei", "Hunan", "Guangdong", "Guangxi", "Hainan", "Chongqing", "Sichuan", "Guizhou", "Yunnan", "Xizang", "Shanxi2", "Gansu", "Qinghai", "Ningxia", "Xinjiang", "child", "teenager", "adult", "middle-ager", "elderly", "breakfast", "lunch", "dinner", "afternoon-tea", "night-snack", "male", "female", "spring", "summer", "autumn", "winter"]

class MealService(Tool):
    def __init__(self, name="餐饮推荐服务", description=mealservice_desc, max_judge=5, topK=5):
        super().__init__(name, description)

        self.items = pd.read_csv('dataset/item.csv')
        self.items = self.items.dropna(axis=0, how='any')
        self.items.set_index(inplace=True, keys=['is_dinner', 'cuisine', 'food_type'])
        self.items['id'] = self.items['city_id'].astype(str) + '_' + self.items['restaurant_id'].astype(str) + '_' + self.items['food_id'].astype(str)
        self.cols = ['is_dinner', 'cuisine', 'food_type']
        self.item_names = self.items['food_name'].to_list()
        self.item_ids = self.items['id'].to_list()

        self.max_judge = max_judge
        self.topK = topK

    def __call__(self, parameter: dict, user_info: list[dict], history: list[dict]):
        output_format = [f"<推荐物品名{i + 1}>" for i in range(self.topK)]
        prompt = recommend_prompt.format(top_k=self.topK, user_info=user_info, dialogue=history, output_format=output_format)
        messages = [{'role': 'user', 'content': prompt}]
        completion = self.invoke_llm(messages)
        match = re.search(r'\s*(\[.*\])\s*', completion, re.DOTALL)

        if match is None:
            return completion, None
        else:
            raw_candidates = eval(match.group(1))

            retrieved_items = self.retrieve(raw_candidates)

            judge_result = self.judge(raw_candidates, retrieved_items)

            to_encode_items = [key for key,val in judge_result.items() if val is None]
            if len(to_encode_items) > 0:
                codes = self.encode(to_encode_items)
            cnt, recommended_list = 0, []
            for raw_candidate, linked_item in judge_result.items():
                if linked_item is not None:
                    recommended_list.append(linked_item)
                else:
                    match = self.KNN(codes[cnt])
                    recommended_list.append({"item_name":match['food_name'], "item_id":match['id']})
                    cnt += 1
            return recommended_list

    def retrieve(self, raw_candidates: list):
        retrieved_items = []
        for raw_candidate in raw_candidates:
            sim = []
            for item_name in self.item_names:
                sim.append(Levenshtein.distance(raw_candidate, item_name))
            sorted_idx = np.array(sim).argsort().tolist()
            for idx in sorted_idx[:self.max_judge]:
                retrieved_items.append({"item_id":self.item_ids[idx],"item_name":self.item_names[idx]})
        df = pd.DataFrame(retrieved_items)
        df = df.drop_duplicates(subset='item_name')
        retrieved_items = df.to_dict(orient='records')
        return retrieved_items

    def judge(self, raw_candidates: list, retrieved_items: list):
        prompt = judge_prompt.format(raw_candidates=raw_candidates, retrieved_items=[retrieved_item['item_name'] for retrieved_item in retrieved_items])
        messages = [{"role":"user", "content":prompt}]
        completion = self.invoke_llm(messages)
        match = re.search(r'\s*(\{.*\})\s*', completion, re.DOTALL)
        judge_result = eval(match.group(1))
        for key, val in judge_result.items():
            if val is not None:
                for item in retrieved_items:
                    if item['item_name'] == val:
                        judge_result[key] = item
                        break
        return judge_result

    def encode(self, raw_candidates: list):
        prompt = encoding_prompt.format(items=raw_candidates)
        messages = [{'role':'user','content':prompt}]
        completion = self.invoke_llm(messages)
        match = re.search(r'\s*(\[.*\])\s*', completion, re.DOTALL)
        return json.loads(match.group(1))

    def KNN(self, query: dict):
        for attr in ['饮食类型', '菜系', '中西餐']:
            if query[attr] == 0:
                query[attr] = slice(None)
        if query['价格'] != '未加限定':
            filtered_items = self.items[
                (self.items['price'] >= query['价格'][0]) & (self.items['price'] <= query['价格'][1])]
        else:
            filtered_items = self.items
        indexes = (query['饮食类型'], query['菜系'], query['中西餐'])
        filtered_items = filtered_items[(filtered_items['not-spicy']==query['不辣'])|
                                        (filtered_items['slightly-spicy']==query['微辣'])|
                                        (filtered_items['medium-spicy']==query['中辣'])|
                                        (filtered_items['extra-spicy']==query['特辣'])]
        filtered_items = filtered_items.loc[indexes]
        vectors = filtered_items[soft_constraints].to_numpy()
        vector = np.array(list(query.values())[8:]).reshape(1,-1)
        sim = cosine_similarity(vector, vectors)
        top_idx = sim.argmax(axis=1)
        return filtered_items.iloc[top_idx.item()]

    def invoke_llm(self, messages: list[dict]) -> str:
        client = OpenAI(
            api_key="your_api_key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion_generator = client.chat.completions.create(
            model="qwen3-235b-a22b",
            messages=messages,
            stop="Observation",
            extra_body={"enable_thinking": False},
            stream=True,
        )
        completion = ''
        for chunk in completion_generator:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                completion += delta.content
        return completion