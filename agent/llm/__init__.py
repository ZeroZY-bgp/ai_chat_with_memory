import openai
from abc import abstractmethod

from transformers import AutoTokenizer, AutoModel


class BaseLLM:
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    model_name = ""

    def chat(self,
             query: str,
             history: list) -> str:
        return self.get_response(query, history)

    @abstractmethod
    def get_response(self, query, history):
        return " "

    @abstractmethod
    def set_para(self, **kwargs):
        pass


class GPT3_5LLM(BaseLLM):
    temperature = 0.1,
    max_token = 1000,
    model_name = 'gpt-3.5-turbo',
    streaming = False

    def __init__(self,
                 temperature=0.1,
                 max_token=1000,
                 model_name='gpt-3.5-turbo',
                 streaming=False):
        self.set_para(temperature, max_token, model_name, streaming)

    def set_para(self,
                 temperature,
                 max_token,
                 model_name,
                 streaming):
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.streaming = streaming

    def send(self, massages):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=massages,
            temperature=self.temperature,
            max_tokens=self.max_token
        ).choices[0].message.content

    def send_stream(self, massages):
        for chunk in openai.ChatCompletion.create(
                model=self.model_name,
                messages=massages,
                temperature=self.temperature,
                max_tokens=self.max_token,
                stream=True,
        ):
            res = chunk["choices"][0].get("delta", {}).get("content")
            if res is not None:
                yield res

    @staticmethod
    def create_massages(query, history):
        # 使用system提示词
        massages = [{'role': 'system', 'content': history[0][0]}]
        for i in range(1, len(history)):
            # 有消息才将信息加入
            if history[i][0] != '':
                massages.append({'role': 'user', 'content': history[i][0]})
            if history[i][1] != '':
                massages.append({'role': 'assistant', 'content': history[i][1]})
        # massages = []
        # for i in range(0, len(history)):
        #     massages.append({'role': 'user', 'content': history[i][0]})
        #     massages.append({'role': 'assistant', 'content': history[i][1]})
        #
        massages.append({'role': 'user', 'content': query})

        return massages

    def get_response(self, query, history):
        massages = self.create_massages(query, history)
        if self.streaming:
            return self.send_stream(massages)
        else:
            return self.send(massages)


class ChatGLMLLM(BaseLLM):
    tokenizer: object = None
    model: object = None

    model_name = 'chatglm-6b-int4'
    device = 'cuda'
    temperature = 0.1,
    max_token = 1000,
    streaming = False

    def __init__(self, temperature=0.1,
                 max_token=1000,
                 model_name='chatglm-6b-int4',
                 streaming=False):
        self.set_para(temperature, max_token, model_name, streaming)

    def set_para(self,
                 temperature,
                 max_token,
                 model_name,
                 streaming):
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.streaming = streaming

    def set_device(self, device):
        self.device = device

    def get_response(self, query, history):
        ans, _ = self.model.chat(
            self.tokenizer,
            query,
            history=history,
            max_length=self.max_token,
            temperature=self.temperature
        )
        return ans

    def change_model_name(self, model_name="chatglm-6b"):
        self.model_name = model_name
        self.load_model(model_name=model_name)

    def load_model(self,
                   model_name: str = "chatglm-6b-int4",
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if self.device == 'cuda':
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).float()
        self.model = self.model.eval()
