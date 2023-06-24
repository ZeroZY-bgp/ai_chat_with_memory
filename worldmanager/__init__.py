import os.path

from tools.generator import DialogEventGenerator
from tools.utils import create_folder, create_txt, create_txt_no_content, CharacterInfo
from template import PROMPT_TEMPLATE, IDENTITY_TEMPLATE


class Manager:

    def __init__(self, world_name):
        self.world_name = world_name
        # 主目录
        self.world_folder = "memory/" + self.world_name
        self.extra_txt = self.world_folder + '/extra.txt'

    def check(self, world_name, ai_name):
        return world_name and ai_name and self.world_is_created(world_name) and self.character_is_created(world_name, ai_name)

    def create_event(self, character_lst):
        """
        :param character_lst: 欲创建事件的角色列表
        """
        if not self.world_is_created(self.world_name):
            print("世界", self.world_name, "未创建，请先调用'create_world(template=False)'函数创建该世界")
            return
        eg = DialogEventGenerator(self.world_name, character_lst)
        eg.do(character_lst)

    def create_world(self, template=False):
        """
        :param template: 是否使用模板，为True则会生成模板人物
        """
        if self.world_is_created(self.world_name):
            print("该世界已存在，请检查世界名称或文件夹")
        else:
            # 添加文件树
            # 主目录
            create_folder(self.world_folder)
            # 全局额外信息记录
            create_txt_no_content(self.extra_txt)
            print("世界已创建")
            # 创建模板
            if template:
                ai_name = '小明'
                # 提示词模板
                prompt_str = PROMPT_TEMPLATE
                # 身份模板
                identity_template = IDENTITY_TEMPLATE.replace("{{{AI_NAME}}}", ai_name)
                info = CharacterInfo(self.world_name, ai_name)
                self.create_character(info, prompt_str, identity_template)
                print("模板人物已创建")

    @staticmethod
    def character_is_created(world_name, ai_name):
        world_folder = "memory/" + world_name
        return os.path.exists(world_folder + '/' + ai_name)

    @staticmethod
    def world_is_created(world_name):
        return os.path.exists("memory/" + world_name)

    def create_character(self, info: CharacterInfo, prompt_str=PROMPT_TEMPLATE, identity_str=IDENTITY_TEMPLATE):
        """
        :param info: 角色信息类（包含世界名称和角色名称）
        :param prompt_str: 提示词
        :param prompt_str: 提示词
        :param identity_str: 身份信息
        """
        if not self.world_is_created(info.world_name):
            print("世界", self.world_name, "未创建，请先调用'create_world'函数创建该世界")
            return False
        # 检查人物是否存在
        if self.character_is_created(info.world_name, info.ai_name):
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
        identity_str = identity_str.replace("{{{AI_NAME}}}", info.ai_name)
        create_txt(info.entity_path, identity_str)
        print("角色", "\"" + info.ai_name + "\"", "已创建")
        return True


if __name__ == '__main__':
    # 初始化世界管理器
    manager = Manager(world_name="test")
    # 创建一个世界，如果已存在则不会创建，template为True则生成模板人物
    manager.create_world(template=True)
    # 角色提示词
    p_str = PROMPT_TEMPLATE
    # 身份信息
    i_str = IDENTITY_TEMPLATE.replace("{{{AI_NAME}}}", "小红")
    information = CharacterInfo(world_name="test", ai_name="小红")
    # 创建角色
    manager.create_character(information, prompt_str=p_str, identity_str=i_str)
