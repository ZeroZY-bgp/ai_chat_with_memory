import os.path

import openai

from agent.chatgpt_tools import EventGenerator
from agent.utils import load_txt_to_str, create_folder, create_txt, create_txt_no_content, append_to_str_file, \
    append_to_lst_file, append_to_dict_file, CharacterInfo
from template import PROMPT_TEMPLATE, IDENTITY_TEMPLATE

openai.api_key = ''


class Manager:
    world_is_created = False
    world_name = ""

    def __init__(self, world_name):
        self.world_name = world_name
        # 主目录
        self.world_folder = "agent/memory/" + self.world_name
        self.global_txt = self.world_folder + '/global.txt'
        self.extra_txt = self.world_folder + '/extra.txt'
        if os.path.exists(self.world_folder):
            self.world_is_created = True

    def create_event(self, character_lst):
        """
        :param character_lst: 欲创建事件的角色列表
        """
        if not self.world_is_created:
            print("世界", self.world_name, "未创建，请先调用'create_world(template=False)'函数创建该世界")
            return
        eg = EventGenerator(self.world_name, character_lst)
        eg.do(character_lst)

    def create_world(self, template=False):
        """
        :param template: 是否使用模板，为True则会生成模板人物
        """
        if self.world_is_created:
            print("该世界已存在，请检查世界名称或文件夹")
        else:
            # 添加文件树
            # 主目录
            create_folder(self.world_folder)
            # 全局信息记录（字典）
            create_txt(self.global_txt, '{}')
            # 全局额外信息记录
            create_txt_no_content(self.extra_txt)
            self.world_is_created = True
            print("世界已创建")
            # 创建模板
            if template:
                ai_name = '小明'
                # 提示词模板
                prompt_str = [('以下是{{{AI_NAME}}}和{{{USER_NAME}}}的对话。'
                               '{{{AI_NAME}}}是一个计算机专业的学生。请大胆猜想他的人设和回答，并补全以下{{{AI_NAME}}}的回答。'
                               '(已知信息):"""{{{context}}}"""\n{{{AI_NAME}}}:',
                               '作为一个计算机专业学生{{{AI_NAME}}}，'
                               '我喜欢人工智能，也喜欢思考和学习。')]
                # 身份模板
                identity_template = '[小明身份]：家里蹲大学计算机专业学生，喜欢编程和学习人工智能，也喜欢思考和学习。'
                info = CharacterInfo(self.world_name, ai_name)
                self.create_character(info, prompt_str, identity_template)
                print("模板人物已创建")

    def character_is_created(self, ai_name):
        return os.path.exists(self.world_folder + '/' + ai_name)

    def create_character(self, info: CharacterInfo, prompt_str=PROMPT_TEMPLATE, identity_str=IDENTITY_TEMPLATE):
        """
        :param info: 角色信息类（包含世界名称和角色名称）
        :param prompt_str: 提示词
        :param prompt_str: 提示词
        :param identity_str: 身份信息
        """
        if not self.world_is_created:
            print("世界", self.world_name, "未创建，请先调用'create_world'函数创建该世界")
            return False
        # 检查人物是否存在
        if self.character_is_created(info.ai_name):
            print("\"" + info.ai_name + "\"" + "已存在")
            return False
        # 人物文件夹
        create_folder(info.folder_path)
        # 提示词
        create_txt(info.prompt_path, str(prompt_str))
        # 事件文件
        create_txt_no_content(info.event_path)
        # 历史对话文件
        create_txt_no_content(info.history_path)
        # 身份信息
        # 1.身份详细信息录入角色本地信息
        identity_str = identity_str.replace("{{{AI_NAME}}}", info.ai_name)
        create_txt(info.identity_path, identity_str)
        # 2.身份文件地址录入全局信息中
        append_to_dict_file(self.global_txt, info.ai_name + '的认知', info.identity_path)
        print("角色", "\"" + info.ai_name + "\"", "已创建")
        return True

    # def create_event(self, character_lst, model_name="gpt3_5"):
    #     eg = EventGenerator(self.world_name, character_lst, model_name=model_name)
    #     eg.do(character_lst)


if __name__ == '__main__':
    # 初始化世界管理器
    manager = Manager(world_name="test")
    # 创建一个世界，如果已存在则不会创建，template为True则生成模板人物
    manager.create_world(template=True)
    # 以下提示词中出现的“{{{}}}”的内容在实际对话中会被替换为具体内容，用户不需要更改
    # 角色提示词
    p_str = [('以下是{{{AI_NAME}}}和{{{USER_NAME}}}的对话。'
              '{{{AI_NAME}}}是一个计算机专业的女生。请大胆猜想她的人设和回答，并补全以下{{{AI_NAME}}}的回答。'
              '(已知信息):"""{{{context}}}"""\n{{{AI_NAME}}}:',
              '作为一个计算机专业学生{{{AI_NAME}}}，'
              '我喜欢人工智能，也喜欢思考和学习。')]
    # 身份信息
    i_str = '[{{{AI_NAME}}}身份]：家外蹲大学计算机专业学生，喜欢编程和学习人工智能，也喜欢思考和学习。'
    information = CharacterInfo(world_name="test", ai_name="小红")
    # 创建角色
    manager.create_character(information, prompt_str=p_str, identity_str=i_str)
    # 产生事件，只有足够的身份信息才能产生具体事件
    cha_lst = ['小明', '小红']
    # manager.create_event(cha_lst, model_name='gpt3_5free')
