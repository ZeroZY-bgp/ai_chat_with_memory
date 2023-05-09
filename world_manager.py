import openai

from agent.chatgpt_tools import EventGenerator
from agent.utils import read_txt_to_str, create_folder, create_txt, create_txt_no_content, append_to_str_file
from template import PROMPT_TEMPLATE, IDENTITY_TEMPLATE

openai.api_key = ''


class Manager:
    world_is_created = False
    world_name = ""

    def __init__(self, world_name):
        self.world_name = world_name
        world_str = read_txt_to_str("agent/memory/world_list.txt")
        world_lst = world_str.split("\n")
        if world_name in world_lst:
            self.world_is_created = True
        # 初始化文件夹树
        self.world_list_file = 'agent/memory/world_list.txt'
        # 主目录
        self.world_folder = "agent/memory/" + self.world_name
        # 提示词
        self.prompt_folder = "agent/memory/" + self.world_name + "/prompts"
        # 世界观和身份记录
        self.global_folder = "agent/memory/" + self.world_name + "/global"
        self.basic_global_file = "agent/memory/" + self.world_name + "/global/all.txt"
        self.character_list_file = "agent/memory/" + self.world_name + "/character.txt"
        # 历史对话记录
        self.local_folder = "agent/memory/" + self.world_name + "/local"

    def create_event(self, character_lst):
        if not self.world_is_created:
            print("世界", self.world_name, "未创建，请先调用'create_world(template=False)'函数创建该世界")
            return
        eg = EventGenerator(self.world_name, character_lst)
        eg.do(character_lst)

    def create_world(self, template=False):
        if self.world_is_created:
            print("该世界已存在，请检查世界名称或文件夹")
        else:
            # 世界列表添加新世界
            append_to_str_file(self.world_list_file, self.world_name + '\n')
            # 添加文件树
            # 主目录
            create_folder(self.world_folder)
            # 创建人物目录记录文件
            create_txt_no_content(self.character_list_file)
            # 提示词
            create_folder(self.prompt_folder)
            # 世界观和身份记录
            create_folder(self.global_folder)
            create_txt_no_content(self.basic_global_file)
            # 历史对话记录
            create_folder(self.local_folder)
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
                self.create_character(ai_name, prompt_str, identity_template)
                print("模板已创建")
            print("世界已创建")

    def create_character(self, ai_name, prompt_str=PROMPT_TEMPLATE, identity_str=IDENTITY_TEMPLATE):
        if not self.world_is_created:
            print("世界", self.world_name, "未创建，请先调用'create_world(template=False)'函数创建该世界")
            return
        # 检查人物是否存在
        cha_str = read_txt_to_str(self.character_list_file)
        cha_lst = cha_str.split("\n")
        if ai_name in cha_lst:
            print("\"" + ai_name + "\"" + "已存在")
            return
        # 添加人物名称到人物列表
        append_to_str_file(self.character_list_file, ai_name + '\n')
        # 提示词
        prompt_file = self.prompt_folder + "/" + ai_name + '.txt'
        create_txt(prompt_file, str(prompt_str))
        # 人物文件夹
        personal_folder = self.local_folder + "/" + ai_name
        create_folder(personal_folder)
        # 事件文件
        event_file_str = personal_folder + "/event.txt"
        create_txt(event_file_str, '[]')
        # 历史对话文件
        history_file_str = personal_folder + "/history.txt"
        create_txt(history_file_str, '[]')
        # 身份信息
        identity_str = identity_str.replace("{{{AI_NAME}}}", ai_name)
        append_to_str_file(self.basic_global_file, identity_str + '\n')
        print("角色", "\"" + ai_name + "\"", "已创建")

    def create_event(self, character_lst, model_name="gpt3_5"):
        eg = EventGenerator(self.world_name, character_lst, model_name=model_name)
        eg.do(character_lst)


if __name__ == '__main__':
    # 初始化世界管理器
    manager = Manager(world_name="A_03")
    # 创建一个世界，如果已存在则不会创建，template为True则生成模板人物
    manager.create_world(template=True)
    # 以下提示词中出现的“{{{}}}”的内容在实际对话中会被替换为具体内容，用户不需要更改
    # 角色提示词
    prompt_str = [('以下是{{{AI_NAME}}}和{{{USER_NAME}}}的对话。'
                   '{{{AI_NAME}}}是一个计算机专业的女生。请大胆猜想她的人设和回答，并补全以下{{{AI_NAME}}}的回答。'
                   '(已知信息):"""{{{context}}}"""\n{{{AI_NAME}}}:',
                   '作为一个计算机专业学生{{{AI_NAME}}}，'
                   '我喜欢人工智能，也喜欢思考和学习。')]
    # 身份信息
    identity_str = '[{{{AI_NAME}}}身份]：家里蹲大学计算机专业学生，喜欢编程和学习人工智能，也喜欢思考和学习。'
    # 创建角色，需要三个必要信息：ai名字，角色提示词，身份信息
    manager.create_character(ai_name="小红", prompt_str=prompt_str, identity_str=IDENTITY_TEMPLATE)

    # 产生事件，只有足够的身份信息才能产生具体事件
    cha_lst = ['小明', '小红']
    manager.create_event(cha_lst, model_name='gpt3_5free')
