import copy
import os
import re

import openai
from abc import abstractmethod

import agent.useless as ul
from transformers import AutoTokenizer, AutoModel

from agent.utils import append_to_str_file, create_txt, load_txt_to_str, load_last_n_lines

DEBUG_MODE = True


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
    window_decrease_size = 400
    model_name = ""
    ai_name = ""
    user_name = ""
    lock_memory = False

    def __init__(self, info, user_name, lock_memory, history_window):
        self.info = info
        self.ai_name = self.info.ai_name
        self.user_name = user_name
        self.lock_memory = lock_memory
        self.history_window = history_window

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def load_history(self, basic_history):
        self.basic_history = basic_history
        self.basic_token_len = len(self.basic_history[0][0]) + len(self.basic_history[0][1])
        self.total_token_size = 0
        if os.path.exists(self.info.history_path) and os.path.getsize(self.info.history_path) == 0:
            # 历史记录为空
            self.history = copy.deepcopy(self.basic_history)
        else:
            self.history = basic_history
            # 加载历史记录最后几行
            history_lst = load_last_n_lines(self.info.history_path, self.history_window)
            pattern = r'(.+?) ' + self.info.ai_name + '说：(.+?)(?=$)'

            for dialog in history_lst:
                matches = re.findall(pattern, dialog)
                self.history.append(matches[0])

    def set_max_history_size(self, max_history_size):
        self.max_history_size = max_history_size

    def chat(self,
             query: str,
             context: str) -> str:
        # 将上下文加入到最开头的提示词中
        context = context.replace("{{{AI_NAME}}}", self.ai_name)
        history_and_context = self.history[0][0].replace("{{{context}}}", context).replace("{{{AI_NAME}}}",
                                                                                           self.ai_name).\
            replace("{{{USER_NAME}}}", self.user_name if self.user_name != '' else '一个人')
        first_ans = self.history[0][1].replace("{{{AI_NAME}}}", self.ai_name)

        if DEBUG_MODE:
            print("context长度:", len(context))
            print("提示词总长度:", len(history_and_context))

        self.history[0] = (history_and_context, first_ans)

        if DEBUG_MODE:
            print("记忆检索片段：")
            print(context)

        ans = self.get_response(query)

        # if DEBUG_MODE and self.model_name == 'gpt-3.5-turbo':
        #
        #     res = openai.Moderation.create(
        #         input=ans
        #     )
        #     print("flag:", res["results"][0])

        self.history.append((query, ans))
        self.total_token_size += (len(ans) + len(query))

        print("Token size:", self.total_token_size + len(context))

        # 恢复最开头的提示词
        self.history[0] = self.basic_history[0]
        # 记录临时历史窗口
        # create_txt(self.tmp_history_path, str(self.history))

        if not self.lock_memory:
            # 保存历史到文件中
            append_str = self.history[-1][0].replace('\n', ' ') + self.history[-1][1] + '\n'
            append_to_str_file(self.info.history_path, append_str)
        # 窗口控制
        self.history_window_control(context)
        return ans

    @abstractmethod
    def get_response(self, query):
        return " "

    def history_window_control(self, context):
        if self.total_token_size + len(context) >= self.max_history_size:
            while self.total_token_size + len(context) > (self.max_history_size - self.window_decrease_size):
                try:
                    self.total_token_size -= (len(self.history[1][0]) + len(self.history[1][1]))
                    self.history.pop(1)
                except IndexError:
                    # print("窗口不能再缩小了")
                    break
            if DEBUG_MODE:
                print("窗口缩小， 历史对话：")
                print(self.history)


class Gpt3_5LLM(BaseLLM):
    model_name = 'gpt-3.5-turbo'
    temperature = 0.1

    def __init__(self,
                 info,
                 user_name,
                 lock_memory=False,
                 history_window=6,
                 temperature=0.1,
                 max_token=1000,
                 window_decrease_size=300):
        super().__init__(info, user_name, lock_memory, history_window)
        self.temperature = temperature
        self.max_token = max_token
        self.window_decrease_size = window_decrease_size

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


class Gpt3_5freeLLM(BaseLLM):
    model_name = 'gpt-3.5-turbo'
    temperature = 0.1

    def __init__(self,
                 info,
                 user_name,
                 lock_memory=False,
                 history_window=6,
                 temperature=0.1,
                 max_token=1000,
                 window_decrease_size=200):
        super().__init__(info, user_name, lock_memory, history_window)
        self.temperature = temperature
        self.max_token = max_token
        self.message_id = ""
        self.talk_times = 0
        self.window_decrease_size = window_decrease_size

    def send(self, prompt, sys_mes):
        return ul.Completion.create(prompt=prompt,
                                    systemMessage=sys_mes,
                                    parentMessageId=self.message_id,
                                    temperature=self.temperature)

    def get_response(self, query):
        res = self.send(query, self.history[0][0])
        self.talk_times += 1
        print("talk_times:", self.talk_times)
        self.message_id = res["id"]
        return res['text']


class ChatGLMLLM(BaseLLM):
    tokenizer: object = None
    model: object = None
    model_name = 'ChatGlm'

    def __init__(self, info, user_name, temperature=0.1, lock_memory=False, history_window=6):
        super().__init__(info, user_name, lock_memory, history_window)
        self.temperature = temperature

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
