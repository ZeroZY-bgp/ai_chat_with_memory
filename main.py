import openai
import configparser
import os

from agent.main_agent import MainAgent
from agent.utils import CharacterInfo
from world_manager import Manager


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


def input_world_name():
    is_created = False
    w_name = ''
    mgr = None
    while not is_created:
        print("输入世界名称：", end=' ')
        w_name = input()
        mgr = Manager(world_name=w_name)
        is_created = mgr.world_is_created
        if not is_created:
            print("世界" + w_name + "不存在，请检查agent/memory内是否存在该世界文件夹，或重新创建世界。")
    return w_name, mgr


def input_ai_name(mgr):
    is_created = False
    name = ''
    while not is_created:
        print("输入ai名称：", end=' ')
        name = input()
        is_created = mgr.character_is_created(name)
        if not is_created:
            print("角色" + name + "不存在，请检查agent/memory/" + world_name +
                  "文件夹内是否存在该角色，或创建该角色。")
    return name


if __name__ == '__main__':

    print("【---欢迎使用ai chat with memory---】")
    while True:
        print("输入数字以选择功能：\n1.与ai对话\n2.管理世界\n3.打开世界文件夹")
        option = input()
        if option == '1':
            print("是否使用config.ini预设参数？y.使用预设参数;其他.手动设置(仅设置世界名称，ai名称和用户名称三项参数)")
            option1 = input()
            if option1 != 'y' and option1 != 'Y':
                world_name, manager = input_world_name()
                ai_name = input_ai_name(manager)
                print("输入用户名称：", end=' ')
                user_name = input()
            else:
                # 检查各项参数是否合法
                manager = Manager(world_name)
                if not manager.world_is_created:
                    print(world_name, "世界未创建，请检查config.ini。")
                    break

            print("设置完毕")
            chat_with_ai(MainAgent(world_name=world_name,
                                   ai_name=ai_name,
                                   user_name=user_name,
                                   model_name=model_name,
                                   lock_memory=lock_memory,
                                   history_window=history_window,
                                   temperature=temperature,
                                   max_history_size=history_window_size))
        elif option == '2':
            print("【---欢迎使用世界管理器---】")
            while True:
                print("你想做什么？\n1.创建新世界；2.创建新人物；3.修改人物信息；")
                option2 = input()
                if option2 == '1':
                    print("输入想创建的世界名称：", end=' ')
                    world_name = input()
                    manager = Manager(world_name)
                    if manager.world_is_created:
                        print("该世界已存在")
                    else:
                        print("是否使用模板？(人物为小明，包括提示词、身份信息。)y.使用模板 其他.不使用模板")
                        y = input()
                        use_template = (y == 'Y' or y == 'y')
                        manager.create_world(use_template)
                        print("是否打开世界文件夹？y.打开 其他.不打开")
                        y = input()
                        if y == 'Y' or y == 'y':
                            path = os.path.abspath('agent\\memory\\' + world_name)
                            os.startfile(path)
                elif option2 == '2':
                    world_name, manager = input_world_name()
                    print("输入要创建的新人物名称：", end=' ')
                    ai_name = input()
                    # 角色提示词
                    prompt_str = [('以下是{{{AI_NAME}}}和{{{USER_NAME}}}的对话。'
                                   '{{{AI_NAME}}}是一个计算机专业的学生。请大胆猜想他/她的人设和回答，并补全以下{{{AI_NAME}}}的回答。'
                                   '(已知信息):"""{{{context}}}"""\n{{{AI_NAME}}}:',
                                   '作为一个计算机专业学生{{{AI_NAME}}}，'
                                   '我喜欢人工智能，也喜欢思考和学习。')]
                    # 身份信息
                    identity_str = '[{{{AI_NAME}}}身份]：家外蹲大学计算机专业学生，喜欢编程和学习人工智能，也喜欢思考和学习。'
                    info = CharacterInfo(world_name, ai_name)
                    if manager.create_character(info, prompt_str, identity_str):
                        print("是否打开世界文件夹？y.打开 其他.不打开")
                        y = input()
                        if y == 'Y' or y == 'y':
                            path = os.path.abspath('agent\\memory\\' + world_name)
                            os.startfile(path)
                elif option2 == '3':
                    world_name, manager = input_world_name()
                    ai_name = input_ai_name(manager)
                    print("当前角色提示词：")

        elif option == '3':
            w_name = input("输入世界名称：")
            path = os.path.abspath('agent\\memory\\' + w_name)
            try:
                os.startfile(path)
            except FileNotFoundError:
                print("没有该世界对应的文件夹，请确认是否拼写正确或重新创建该世界。")
        else:
            print("请输入正确数字(1-3)。")
