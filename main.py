from agent.main_agent import MainAgent
from agent.chatgpt_tools import EventGenerator


def chat_with_ai():
    agent = MainAgent(world_name="A_02",
                      ai_name='Alice',
                      model_name='gpt3_5',
                      lock_memory=True,
                      lock_event=True,
                      classifier_enabled=False,
                      max_history_size=1600)

    print("---初始化完成，对话开始---")
    while True:
        chat_str = input()
        if chat_str == 'exit':
            break
        agent.chat(chat_str)


def create_event():
    character_lst = ['Alice', 'Lisa']
    eg = EventGenerator(world_name="A_02", character_lst=character_lst)
    eg.do(character_lst)


if __name__ == '__main__':
    chat_with_ai()
    # create_event()
