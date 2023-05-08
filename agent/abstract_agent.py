from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    def __init__(self):
        self.ai_name = ''

    @abstractmethod
    def chat(self, string):
        pass
