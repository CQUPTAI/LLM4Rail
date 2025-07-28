import os
from openai import OpenAI

class UserSimulator:
    def __init__(self):
        self.base_prompt = \
'''你正在与推荐系统对话来寻求推荐。你的目标物品是：{target_item}。你在对话时必须遵循以下命令：
如果推荐系统为你推荐了与{target_item}相同的食品或饮品，你应当回复且仅回复"<推荐列表中所接受的物品名>这正是我想要的，谢谢你！"；
如果推荐系统为你推荐了除{target_item}以外的其它物品，你应当表示拒绝并提供关于{target_item}的信息。注意：你绝不可以说出目标物品的名称！！
如果推荐系统询问你的偏好，你应当提供关于{target_item}的信息。注意：你绝不可以说出目标物品的名称！！
下面是对话历史：
'''

    def set_target(self, target_item):
        self.target_item = target_item

    def __call__(self, history: list[dict]):    # Python支持怎样的数据类型表达方法？
        completion = self.invoke_llm(history)
        return completion

    def invoke_llm(self, history: list[dict]):
        prompt = self.base_prompt.format(target_item=self.target_item)
        for turn in history:
            role = '你' if turn['role']=='seeker' else '推荐系统'
            prompt += f"{role}: {turn['content']}\n"
        prompt += "你: "
        messages = [{'role': 'user', 'content': prompt}]
        client = OpenAI(
            api_key=os.environ.get('QwenKey', ''),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=messages,
        )
        return completion.choices[0].message.content
