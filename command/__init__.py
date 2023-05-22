import configparser
import os

from tools import DialogEventGenerator
from tools.utils import load_txt_to_str, append_to_str_file, load_last_n_lines, VectorStore, load_txt_to_lst

command_config = configparser.ConfigParser()
command_config.read('command/command.ini', encoding='utf-8-sig')


# 指令执行记号，是一个全局变量，用于作用到整个系统
class CommandFlags:
    def __init__(self):
        self.reset()

    def reset(self):
        self.not_command = True
        self.wrong_command = False
        self.open = False

        self.history = False
        self.prompt = False
        self.event = False
        self.entity = False
        self.folder = False
        self.show_temp_history = False
        self.show_prompt = False
        self.show_context = False

        self.dialog_to_event = False
        self.continue_talk = False
        self.retry = False
        self.help = False
        self.exit = False
        self.user_name = ''
        self.ai_name = ''


command_flags = CommandFlags()
command_flags.reset()


class Pool:

    def __init__(self):
        self.command_start = command_config.get('START', 'command_start')

    def check(self, command: str, ai_name):
        """
        :param ai_name: 执行指令时要回答的ai
        :param command: 指令字符串
        :return: 指令执行结果，用于与调用方通信
        """
        if command[0] != self.command_start:
            return ''
        command_flags.not_command = False
        if len(command) == 1:
            print("（您输入了空指令，输入'" + self.command_start + "help'来获取帮助。'）")
            command_flags.wrong_command = True
            return 'wrong command'
        if command[1:] not in command_config['LIST'].values():
            print("（指令不存在，输入'" + self.command_start + "help'来获取帮助。'）")
            command_flags.wrong_command = True
            return 'wrong command'

        command_flags.ai_name = ai_name
        # command_flags.user_name = var_dict.user_name
        command_list = command_config['LIST']
        # 以下是指令处理部分，该部分可以自定义指令
        # 打开对话历史文件
        command = command[1:]
        # 打开文件操作暂时只支持Windows系统
        if command == command_list['history']:
            command_flags.history = True
            command_flags.open = True
        elif command == command_list['prompt']:
            command_flags.prompt = True
            command_flags.open = True
        elif command == command_list['event']:
            command_flags.event = True
            command_flags.open = True
        elif command == command_list['entity']:
            command_flags.entity = True
            command_flags.open = True
        elif command == command_list['folder']:
            command_flags.folder = True
            command_flags.open = True
        elif command == command_list['dialog_to_event']:
            command_flags.dialog_to_event = True
            return 'dialog to event'
        elif command == command_list['continue_talk']:
            command_flags.continue_talk = True
            return 'dialog to event'
        elif command == command_list['retry']:
            command_flags.retry = True
            return 'retry'
        elif command == command_list['help']:
            # help文档
            path = command_config['HELP']['info_path']
            command_flags.help = True
            print(load_txt_to_str(path))
            return 'help'
        elif command == command_list['exit']:
            # 退出
            command_flags.exit = True
            return 'exit'
        elif command == command_list['show_context']:
            # 展示临时窗口历史对话
            command_flags.show_context = True
            return 'show_context'
        elif command == command_list['show_temp_history']:
            # 展示临时窗口历史对话
            command_flags.show_temp_history = True
            return 'show_temp_history'
        elif command == command_list['show_prompt']:
            # 展示临时窗口历史对话
            command_flags.show_prompt = True
            return 'show_prompt'
        else:
            print("（指令不存在，输入'" + self.command_start + "help'来获取帮助。'）")
            command_flags.wrong_command = True
        return 'no process'


def open_command(agent):
    if command_flags.history:
        path = agent.info.history_path
    elif command_flags.prompt:
        path = agent.info.prompt_path
    elif command_flags.event:
        path = agent.info.event_path
    elif command_flags.entity:
        path = agent.info.entity_path
    elif command_flags.folder:
        path = agent.info.folder_path
    else:
        return
    try:
        path = os.path.abspath(path)
        os.startfile(path)
    except AttributeError:
        print("文件未能成功打开。")


def show_command(agent):
    if command_flags.show_temp_history:
        # 展示当前临时历史窗口
        window = agent.history
        for dialog in window[1:]:
            print(dialog[0])
            print(dialog[1])
    elif command_flags.show_prompt:
        # 展示当前提示词
        print("提示词：")
        print(agent.cur_prompt)
    elif command_flags.show_context:
        print("实体记忆：")
        print(agent.entity_text)
        print("对话记忆：")
        print(agent.dialog_text)
        print("事件记忆：")
        print(agent.event_text)


def execute_command(agent):
    if command_flags.open:
        open_command(agent)
        return 'ai_chat_with_memory sys:open'
    elif command_flags.dialog_to_event:
        history_window_size = input("输入要转换的对话窗口大小：")
        if not history_window_size.isdigit():
            print("非法数字")
            return 'ai_chat_with_memory sys:not number'
        # 提取历史记录
        history_lst = load_last_n_lines(agent.info.history_path, int(history_window_size))
        history_str = ''
        for dialog in history_lst:
            history_str += dialog
            print(dialog)
        option = input("转换以上对话为新事件？y.确定；其他.取消")
        if option == 'y' or option == 'Y':

            deg = DialogEventGenerator()
            while True:
                print("正在生成中...")
                event_str = deg.do(history_str)
                print("新事件：")
                print(event_str)
                option1 = input("加入该事件？y.确定；r.基于当前窗口重新生成；其他.取消生成")
                if option1 == 'y' or option1 == 'Y':
                    append_to_str_file(agent.info.event_path, event_str + '\n')
                    command_flags.event = True
                    return
                elif option1 == 'r' or option1 == 'R':
                    continue
                else:
                    return
    elif command_flags.continue_talk:
        return 'ai_chat_with_memory sys:continue'
    elif command_flags.exit:
        # 执行退出指令
        return 'ai_chat_with_memory sys:exit'
    # 若query是一个非重试指令，则处理过后退出，不进行对话
    if not command_flags.not_command and not command_flags.retry:
        show_command(agent)
        return 'ai_chat_with_memory sys:执行了指令'
    return ''


def command_cleanup_task(agent):
    if command_flags.ai_name != agent.ai_name:
        return
    if command_flags.history:
        if agent.lock_memory:
            # 历史对话被打开过，重新加载历史对话（仅当lock_memory为True时重新加载）
            agent.history_vs = VectorStore(agent.embeddings, agent.info.history_path, chunk_size=1,
                                           top_k=agent.history_top_k)
        # 重新加载临时历史对话
        agent.load_history(agent.basic_history)
    elif command_flags.prompt:
        # 提示词被打开过，重新加载提示词和历史对话
        agent.basic_history = load_txt_to_lst(agent.info.prompt_path)
        agent.load_history(agent.basic_history)
    elif command_flags.event:
        # 事件被打开过，重新加载事件
        agent.event_vs = VectorStore(agent.embeddings, agent.info.event_path, chunk_size=30,
                                     top_k=agent.event_top_k)
    elif command_flags.entity:
        agent.entity_vs = VectorStore(agent.embeddings, agent.info.entity_path, chunk_size=30,
                                      top_k=agent.entity_top_k)
    command_flags.reset()
