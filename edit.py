import sys
import os

from tools.utils import CharacterInfo
from template import PROMPT_TEMPLATE, IDENTITY_TEMPLATE
from worldmanager import Manager
from worldsandbox import Sandbox
from config import BaseConfig


if __name__ == '__main__':

    print("【---欢迎使用AI chat with memory---】")
    while True:
        # print("输入数字以选择功能：\n1.与ai对话\n2.ai之间对话\n3.管理世界\n4.打开世界文件夹\n5.打开设置")
        print("输入数字以选择功能：\n1.管理世界\n2.打开世界文件夹\n3.打开设置")
        option = input()
        # if option == '1':
        #     config = BaseConfig()
        #     # 检查参数是否合法
        #     world_name = config.world_name
        #     ai_name = config.ai_name
        #     manager = Manager(world_name)
        #     if not manager.world_is_created:
        #         print(world_name, "世界未创建，请检查config.ini文件。")
        #         break
        #     if not manager.character_is_created(ai_name):
        #         print(ai_name, "人物未创建，请检查config.ini文件。")
        #         break
        #
        #     print("设置完毕")
        #     Sandbox(world_name).chat_with_one_agent(config)
        #     sys.exit(0)
        # elif option == '2':
        #     config = BaseConfig()
        #     # 检查参数是否合法
        #     world_name = config.world_name
        #     ai_name = config.ai_name
        #     manager = Manager(world_name)
        #     if not manager.world_is_created:
        #         print(world_name, "世界未创建，请检查config.ini文件。")
        #         break
        #
        #     is_created = True
        #     for name in config.ai_names:
        #         if not manager.character_is_created(name):
        #             print(name, "人物未创建，请检查config.ini文件。")
        #             is_created = False
        #             break
        #     if not is_created:
        #         break
        #     if len(config.ai_names) == 0:
        #         print("空ai列表，请检查config.ini文件。")
        #         break
        #     print("设置完毕")
        #     Sandbox(world_name).chat_with_multi_agent(config.ai_names, config)

        if option == '1':
            print("【---欢迎使用世界管理器---】")
            while True:
                print("你想做什么？\n1.创建新世界；2.创建新人物；3.返回")
                option2 = input()
                if option2 == '1':
                    print("输入想创建的世界名称：", end=' ')
                    world_name = input()
                    manager = Manager(world_name)
                    if manager.world_is_created(world_name):
                        print("该世界已存在")
                    else:
                        print("是否使用模板？(人物为小明，包括提示词、身份信息。)y.使用模板 其他.不使用模板")
                        y = input()
                        use_template = (y == 'Y' or y == 'y')
                        manager.create_world(use_template)
                        print("是否打开世界文件夹？y.打开 其他.不打开")
                        y = input()
                        if y == 'Y' or y == 'y':
                            path = os.path.abspath('memory\\' + world_name)
                            os.startfile(path)
                elif option2 == '2':
                    world_name = input("输入世界名称：")
                    manager = Manager(world_name)
                    if not manager.world_is_created:
                        print("世界" + world_name + "不存在，请检查输入的世界名称或创建该世界。")
                        break
                    ai_name = input("输入要创建的新人物名称：")
                    if manager.character_is_created(world_name, ai_name):
                        print("人物" + ai_name + "已存在，创建操作已取消。")
                        break
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
                    break
                else:
                    print("请输入正确的数字(1-3)")
        elif option == '2':
            w_name = input("输入世界名称：")
            path = os.path.abspath('memory\\' + w_name)
            try:
                os.startfile(path)
            except FileNotFoundError:
                print("没有该世界对应的文件夹，请确认是否拼写正确。")
        elif option == '3':
            path = os.path.abspath('config.ini')
            os.startfile(path)
        else:
            print("请输入正确数字(1-3)。")
