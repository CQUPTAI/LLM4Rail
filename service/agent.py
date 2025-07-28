import re, os
from openai import OpenAI
from service.modules.meal_service import MealService
from service.modules.ticket_query import TicketQuery
from service.modules.weather_query import WeatherQuery

class Agent():
    def __init__(self, model):
        self.model = model
        self.ticket_query = TicketQuery()
        self.weather_query = WeatherQuery()
        self.meal_service = MealService()
        self.tools = {self.meal_service.name: self.meal_service,
                      self.ticket_query.name: self.ticket_query,
                      self.weather_query.name: self.weather_query}
        self.tool_descriptions = '\n'.join([f'{i+1}.'+tool.description for i, tool in enumerate(self.tools.values())])
        self.tool_names = [tool_name for tool_name in self.tools.keys()]

    def query(self, user_info: dict, history: list, query: str, trajectory: dict) -> str:
        base_prompt = \
f'''Answer the question below as best as you can, you can use the historical information above or access the following tools:
tools: 
{self.tool_descriptions}
user information:
{user_info}

Use the following format to answer:
Question: the question you need to answer
Thought: thinking process about how to answer the question
Action: tool to use for answering, must be one of {self.tool_names}; if any of the arguments is missing, don't take that action.
Action_Input: information needed for taking the action
Observation: result of the action
... (Thought/Action/Action_Input/Observation process can repeat N times)
Thought: I now know the answer. (if the answer is gotten, then output this sentence)
Answer: the final answer to the question

Begin!
'''
        completion = ''
        prompt = base_prompt + f"Question: {query}"
        while(True):
            new_completion = ''
            for chunk in self.invoke_llm(prompt+completion):
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    new_completion += delta.content
            if 'Answer:' in new_completion:
                trajectory['trajectory'].append(new_completion)
                break
            trajectory['trajectory'].append({'completion': new_completion})
            action = re.search(r'Action:\s*(.+)\n', new_completion).group(1).strip()
            action_input = re.search(r'Action_Input:\s*(.+)\n?', new_completion).group(1).strip()
            tool = self.tools[action]
            tool_input = eval(action_input.replace("'", '"')) if action_input != 'None' else None
            obs = tool(tool_input, user_info, history)
            str_obs = f'\nObservation:{obs}\n'
            new_completion += str_obs
            completion += new_completion
            trajectory['trajectory'][-1]['action'] = action
            trajectory['trajectory'][-1]['action_input'] = tool_input
            trajectory['trajectory'][-1]['observation'] = str_obs

    def invoke_llm(self, prompt: str) -> str:
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
        messages = [{'role': 'user', 'content': prompt}]
        params = {"model": model,
                  "messages":messages,
                  "stop":"Observation",
                  "stream":True}
        if self.model == 'qwen-3':
            params["extra_body"] = {"enable_thinking": False}
        completion = client.chat.completions.create(**params)

        return completion
