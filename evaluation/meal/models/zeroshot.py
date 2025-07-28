import re, os
import numpy as np
import pandas as pd
import Levenshtein
from openai import OpenAI

recommend_prompt = \
'''你是一个饮食推荐家，你正在与用户对话为其提供饮食推荐。你在对话时必须遵循以下命令：
如果你对于用户的偏好了解不足，你应当询问用户的偏好；
如果你已经了解用户偏好并能为其推荐饮食，你应当给出推荐。推荐列表必须包括 {top_k} 个食品或饮品，它们要符合用户的偏好和要求。注意：这种情况下你必须输出一个由推荐物品名组成的Python列表：{output_format}！
下面是对话历史：
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

soft_constraints = ["Beijing", "Tianjin", "Hebei", "Shanxi", "Nei Mongol", "Liaoning", "Jilin", "Heilongjiang", "Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei", "Hunan", "Guangdong", "Guangxi", "Hainan", "Chongqing", "Sichuan", "Guizhou", "Yunnan", "Xizang", "Shanxi2", "Gansu", "Qinghai", "Ningxia", "Xinjiang", "child", "teenager", "adult", "middle-ager", "elderly", "breakfast", "lunch", "dinner", "afternoon-tea", "night-snack", "male", "female", "spring", "summer", "autumn", "winter"]

class ZeroShot_CRS():
    def __init__(self, model, max_judge=5, topK=5):
        self.model = model

        self.items = pd.read_csv('dataset/item.csv')
        self.items.set_index(inplace=True, keys=['is_dinner', 'cuisine', 'food_type'])
        self.items['id'] = self.items['city_id'].astype(str) + '_' + self.items['restaurant_id'].astype(str) + '_' + self.items['food_id'].astype(str)
        self.cols = ['is_dinner', 'cuisine', 'food_type']
        self.item_names = self.items['food_name'].to_list()

        self.max_judge = max_judge
        self.topK = topK

    def __call__(self, history: list[dict]):
        output_format = [f"<推荐物品名{i+1}>" for i in range(self.topK)]
        prompt = recommend_prompt.format(top_k=self.topK, output_format=output_format)
        for turn in history:
            prompt += f"{turn['role']}: {turn['content']}\n"
        messages = [{'role': 'user', 'content': prompt}]
        completion = self.invoke_llm(messages)
        match = re.search(r'\s*(\[.*\])\s*', completion, re.DOTALL)

        if match is None:
            return completion, None
        else:
            raw_candidates = eval(match.group(1))
            retrieved_items = self.retrieve(raw_candidates)
            judge_result = self.judge(raw_candidates, retrieved_items)
            recommended_list = [val for val in judge_result.values()]
            return f"我向你推荐：{','.join([item for item in recommended_list if item is not None])}.", recommended_list

    def retrieve(self, raw_candidates: list):
        retrieved_items = []
        for raw_candidate in raw_candidates:
            sim = []
            for item_name in self.item_names:
                sim.append(Levenshtein.distance(raw_candidate, item_name))
            sorted_idx = np.array(sim).argsort().tolist()
            for idx in sorted_idx[:self.max_judge]:
                retrieved_items.append(self.item_names[idx])
        return list(set(retrieved_items))

    def judge(self, raw_candidates: list, retrieved_items: list):
        prompt = judge_prompt.format(raw_candidates=raw_candidates, retrieved_items=retrieved_items)
        messages = [{"role":"user", "content":prompt}]
        completion = self.invoke_llm(messages)
        match = re.search(r'\s*(\{.*\})\s*', completion, re.DOTALL)
        judge_result = eval(match.group(1))
        return judge_result

    def invoke_llm(self, messages: list[dict]) -> str:
        if self.model == 'qwen-3':
            model = 'qwen3-235b-a22b'
            base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            api_key = os.environ.get('QwenKey', '')
        elif self.model == 'gpt-4o':
            model = 'gpt-4o'
            base_url = 'https://api.openai.com/v1'
            api_key = os.environ.get('GPTKey', '')
        else:
            raise Exception('Invalid model name!')
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        params = {"model": model,
                  "messages":messages,
                  "stop":"Observation",
                  "stream":True}
        if self.model == 'qwen-3':
            params["extra_body"] = {"enable_thinking": False}
        completion_generator = client.chat.completions.create(**params)
        completion = ''
        for chunk in completion_generator:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                completion += delta.content
        return completion

