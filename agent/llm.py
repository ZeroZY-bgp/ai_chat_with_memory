import copy
from abc import abstractmethod

import openai
from transformers import AutoTokenizer, AutoModel

from agent.chatgpt_tools import EventDetector
from agent.utils import append_to_lst_file

openai.api_key = ''


class BaseLLM:
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    basic_history = []
    history_len: int = 10
    basic_token_len = 0
    total_token_size = 0
    max_history_size = 600
    model_name = ""
    ai_name = ""
    history_path = ''
    lock_memory = False
    # 事件识别器
    event_det = None

    def __init__(self, ai_name, lock_memory, world_name):
        self.ai_name = ai_name
        self.lock_memory = lock_memory
        self.history_path = 'agent/memory/' + world_name + '/对话/' + self.ai_name + '/history.txt'
        self.event_det = EventDetector()

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def set_history_path(self, path):
        self.history_path = path

    def load_basic_history(self, basic_history):
        self.basic_history = basic_history
        self.history = copy.deepcopy(self.basic_history)
        self.basic_token_len = len(self.basic_history[0][0]) + len(self.basic_history[0][1])
        self.total_token_size = self.basic_token_len

    def set_max_history_size(self, max_history_size):
        self.max_history_size = max_history_size

    def chat(self,
             query: str,
             context: str) -> str:
        # 将上下文加入到最开头的提示词中
        context = context.replace("{{{AI_NAME}}}", self.ai_name)
        init_str = self.history[0][0].replace("{{{context}}}", context).replace("{{{AI_NAME}}}", self.ai_name)
        init_str2 = self.history[0][1].replace("{{{AI_NAME}}}", self.ai_name)
        self.history[0] = (init_str, init_str2)
        print(self.history[0][0])
        ans = self.get_response(query)
        self.history.append((query, ans))
        self.total_token_size += (len(ans) + len(query))
        print("Token size:", self.total_token_size + len(context))
        # 窗口控制
        self.history_window_control(context)
        # 恢复最开头的提示词
        self.history[0] = self.basic_history[0]
        if not self.lock_memory:
            # 保存历史到文件中
            append_to_lst_file(self.history_path, self.history[-1])
        return ans

    @abstractmethod
    def get_response(self, query):
        return " "

    def history_window_control(self, context):
        if self.total_token_size + len(context) >= self.max_history_size:
            while self.total_token_size > (self.max_history_size - 400):
                self.total_token_size -= (len(self.history[1][0]) + len(self.history[1][1]))
                self.history.pop(1)
            print("窗口缩小， 历史对话：")
            print(self.history)

    def _call(self, **kwargs):
        pass


class Gpt3_5LLM(BaseLLM):
    model_name = 'gpt-3.5-turbo'
    temperature = 0.1

    def __init__(self, ai_name, world_name, lock_memory=False, temperature=0.1, max_token=1000):
        super().__init__(ai_name, lock_memory, world_name)
        self.temperature = temperature
        self.max_token = max_token

    def send(self, massages):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=massages,
            temperature=self.temperature,
            max_tokens=self.max_token
        )

    def create_massages(self, query):
        massages = []
        for i in range(len(self.history)):
            massages.append({'role': 'user', 'content': self.history[i][0]})
            massages.append({'role': 'assistant', 'content': self.history[i][1]})

        massages.append({'role': 'user', 'content': query})

        return massages

    def get_response(self, query):
        massages = self.create_massages(query)
        response = self.send(massages)
        return response.choices[0].message.content


class ChatGLMLLM(BaseLLM):
    tokenizer: object = None
    model: object = None
    model_name = 'ChatGlm'

    def __init__(self, ai_name, world_name, lock_memory=False):
        super().__init__(ai_name, lock_memory, world_name)

    def get_response(self, query):
        ans, _ = self.model.chat(
            self.tokenizer,
            query,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        return ans

    def load_model(self,
                   model_name_or_path: str = "chatglm-6b-int4",
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
