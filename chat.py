import openai

from agent.main_agent import MainAgent

openai.api_key = ''


LOCK_MEMORY = True

AI_NAME = '小明'

WORLD_NAME = 'A_03'

USER_NAME = 'user'

HISTORY_WINDOW_SIZE = 1600

TEMPERATURE = 0.6

MODEL_NAME = 'gpt3_5free'


def chat_with_ai():
    agent = MainAgent(world_name=WORLD_NAME,
                      ai_name=AI_NAME,
                      model_name=MODEL_NAME,
                      lock_memory=LOCK_MEMORY,
                      temperature=TEMPERATURE,
                      max_history_size=HISTORY_WINDOW_SIZE)

    print("---初始化完成，对话开始---")
    while True:
        chat_str = input()
        if chat_str == 'exit':
            break
        agent.chat(chat_str)


if __name__ == '__main__':
    chat_with_ai()
