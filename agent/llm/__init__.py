import openai
from abc import abstractmethod

from transformers import AutoTokenizer, AutoModel


class BaseLLM:
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    model_name = ""

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def chat(self,
             query: str,
             history: list) -> str:
        return self.get_response(query, history)

    @abstractmethod
    def get_response(self, query, history):
        return " "


class Gpt3_5LLM(BaseLLM):
    model_name = 'gpt-3.5-turbo-0301'
    temperature = 0.1

    def __init__(self,
                 temperature=0.1,
                 max_token=1000,
                 window_decrease_size=300):
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

    @staticmethod
    def create_massages(query, history):
        massages = []
        for i in range(len(history)):
            massages.append({'role': 'user', 'content': history[i][0]})
            massages.append({'role': 'assistant', 'content': history[i][1]})

        massages.append({'role': 'user', 'content': query})

        return massages

    def get_response(self, query, history):
        massages = self.create_massages(query, history)
        response = self.send(massages)
        return response.choices[0].message.content


class ChatGLMLLM(BaseLLM):
    tokenizer: object = None
    model: object = None
    model_name = 'chatglm-6b-int4'
    device = 'cuda'

    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def set_device(self, device):
        self.device = device

    def get_response(self, query, history):
        ans, _ = self.model.chat(
            self.tokenizer,
            query,
            history=history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        return ans

    def change_model_name(self, model_name="chatglm-6b"):
        self.model_name = model_name
        self.load_model(model_name_or_path=model_name)

    def load_model(self,
                   model_name_or_path: str = "chatglm-6b-int4",
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if self.device == 'cuda':
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float()
        self.model = self.model.eval()
