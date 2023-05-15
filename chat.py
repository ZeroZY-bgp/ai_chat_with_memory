import configparser
import openai
import time

from agent.main_agent import MainAgent

openai.api_key = ''

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8-sig')

openai.api_key = config.get('API', 'openai_api_key')
lock_memory = config.getboolean('MEMORY', 'lock_memory')
history_window = config.getint('MEMORY', 'history_window')
ai_name = config.get('AI', 'name')
world_name = config.get('WORLD', 'name')
user_name = config.get('USER', 'name')
history_window_size = config.getint('HISTORY', 'window_size')
temperature = config.getfloat('TEMPERATURE', 'temperature')
model_name = config.get('MODEL', 'name')


def chat_with_ai(agent):
    print("---初始化完成，对话开始---")
    while True:
        chat_str = input()
        if chat_str == 'exit':
            break
        agent.chat(chat_str)


def ai_vs_ai(agent1, agent2):
    print("---初始化完成，对话开始---")
    chat_str = '现在' + agent2.ai_name + '在你面前，和' + agent2.ai_name + '打个招呼吧。'
    agent1.user_name = 'user'
    chat_str = agent1.chat(chat_str)
    time.sleep(20)
    agent1.user_name = agent2.ai_name
    while True:
        chat_str = agent2.chat(chat_str)
        time.sleep(20)
        if 'exit' in chat_str:
            agent1.chat(chat_str)
            break
        chat_str = agent1.chat(chat_str)
        time.sleep(20)
        if 'exit' in chat_str:
            agent2.chat(chat_str)
            break


if __name__ == '__main__':
    # chat_with_ai(MainAgent(world_name=world_name,
    #                        ai_name=ai_name,
    #                        user_name=user_name,
    #                        model_name=model_name,
    #                        lock_memory=lock_memory,
    #                        temperature=temperature,
    #                        max_history_size=history_window_size))
    ai_name1 = '小明'
    ai_name2 = '小红'
    agent1 = MainAgent(world_name=world_name,
                       ai_name=ai_name1,
                       user_name=ai_name2,
                       model_name=model_name,
                       lock_memory=lock_memory,
                       temperature=temperature,
                       max_history_size=history_window_size)
    agent2 = MainAgent(world_name=world_name,
                       ai_name=ai_name2,
                       user_name=ai_name1,
                       model_name=model_name,
                       lock_memory=lock_memory,
                       temperature=temperature,
                       max_history_size=history_window_size)
    ai_vs_ai(agent1, agent2)
