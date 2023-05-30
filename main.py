import sys

import openai
import configparser
import os

from tools.utils import CharacterInfo
from template import PROMPT_TEMPLATE, IDENTITY_TEMPLATE
from world_manager import Manager
from worldsandbox import Sandbox

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8-sig')


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
                print(ai_name, "人物未创建，请检查config.ini文件。")
                break

            print("设置完毕")
            Sandbox(world_name).chat_with_one_agent(config)
            sys.exit(0)
        elif option == '2':
            config = configparser.ConfigParser()
            config.read('config.ini', encoding='utf-8-sig')
            openai.api_key = config.get('API', 'openai_api_key')
            # 检查参数是否合法
            world_name = config['WORLD']['name']
            ai_names = config['MULTI_AI']['names']
            # 获取名字列表
            valid_punctuation = ['、', '，', '.', '。', '|', '/']
            for p in valid_punctuation:
                ai_names = ai_names.replace(p, ',')
            ai_names = ai_names.split(',')
            ai_names = [name for name in ai_names if name]

            manager = Manager(world_name)
            if not manager.world_is_created:
                print(world_name, "世界未创建，请检查config.ini文件。")
                break

            is_created = True
            for name in ai_names:
                if not manager.character_is_created(name):
                    print(name, "人物未创建，请检查config.ini文件。")
                    is_created = False
                    break
            if not is_created:
                break
            print("设置完毕")
            Sandbox(world_name).chat_with_multi_agent(ai_names, config)

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
