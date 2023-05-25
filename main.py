import importlib
import sys

import openai
import configparser
import os

from langchain.embeddings import HuggingFaceEmbeddings

from agent import MainAgent
from tools.utils import CharacterInfo
from template import PROMPT_TEMPLATE, IDENTITY_TEMPLATE
from world_manager import Manager
from worldsandbox import Sandbox

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8-sig')


def chat_with_ai(agent):
    print("---初始化完成，对话开始---")
    print("'输入/help'可获取指令列表")
    while True:
        chat_str = ''
        while chat_str == '':
            chat_str = input((agent.user_name + "：") if agent.user_name != '' else 'user：')
        back_msg = agent.chat(chat_str)
        if back_msg == 'ai_chat_with_memory sys:exit':
            return


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


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_llm(llm_class_name, temperature):
    llm_class = get_class("agent.llm", llm_class_name)
    llm_instance = llm_class(temperature)
    if hasattr(llm_instance, 'load_model'):
        llm_instance.load_model()
    if hasattr(llm_instance, 'set_device'):
        llm_instance.set_device(config['MODEL']['model_device'])
    return llm_instance


def create_embedding_model(embed_model_path, embed_device):
    return HuggingFaceEmbeddings(model_name=embed_model_path,
                                 model_kwargs={'device': embed_device})


if __name__ == '__main__':

    print("【---欢迎使用ai chat with memory---】")
    while True:
        print("输入数字以选择功能：\n1.与ai对话\n2.ai之间对话\n3.管理世界\n4.打开世界文件夹\n5.打开设置")
        option = input()
        if option == '1':
            config = configparser.ConfigParser()
            config.read('config.ini', encoding='utf-8-sig')
            openai.api_key = config.get('API', 'openai_api_key')
            # 检查参数是否合法
            world_name = config['WORLD']['name']
            ai_name = config['AI']['name']
            manager = Manager(world_name)
            if not manager.world_is_created:
                print(world_name, "世界未创建，请检查config.ini文件。")
                break
            if not manager.character_is_created(ai_name):
                print(world_name, "人物未创建，请检查config.ini文件。")
                break

            print("设置完毕")
            language_model = create_llm(config['MODEL']['name'], config.getfloat('MODEL', 'temperature'))
            embedding_model = create_embedding_model(config['MODEL']['embed_model'],
                                                     config['MODEL']['embed_model_device'])
            agent = MainAgent(llm=language_model,
                              embed_model=embedding_model,
                              config=config)
            Sandbox(world_name).chat_with_one_agent(agent)
            sys.exit(0)
        elif option == '3':
            print("【---欢迎使用世界管理器---】")
            while True:
                print("你想做什么？\n1.创建新世界；2.创建新人物；3.修改人物信息；4.返回")
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
                    prompt_str = PROMPT_TEMPLATE
                    # 身份信息
                    identity_str = IDENTITY_TEMPLATE.replace("{{{AI_NAME}}}", ai_name)
                    info = CharacterInfo(world_name, ai_name)
                    if manager.create_character(info, prompt_str, identity_str):
                        print("是否打开人物文件夹？y.打开 其他.不打开")
                        y = input()
                        if y == 'Y' or y == 'y':
                            path = os.path.abspath(info.folder_path)
                            os.startfile(path)
                elif option2 == '3':
                    world_name, manager = input_world_name()
                    ai_name = input_ai_name(manager)
                    print("当前角色提示词：")
                elif option2 == '4':
                    break
                else:
                    print("请输入正确的数字(1-4)")
        elif option == '4':
            w_name = input("输入世界名称：")
            path = os.path.abspath('agent\\memory\\' + w_name)
            try:
                os.startfile(path)
            except FileNotFoundError:
                print("没有该世界对应的文件夹，请确认是否拼写正确或重新创建该世界。")
        elif option == '5':
            path = os.path.abspath('config.ini')
            os.startfile(path)
        else:
            print("请输入正确数字(1-3)。")
