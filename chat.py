import openai

from agent.main_agent import MainAgent

openai.api_key = ''


WORLD_NAME = 'A_03'

USER_NAME = '用户'

AI_NAME = '小红'

LOCK_MEMORY = False

TEMPERATURE = 0.6

HISTORY_WINDOW_SIZE = 1600

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
