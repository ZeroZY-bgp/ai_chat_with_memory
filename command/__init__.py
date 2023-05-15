import configparser
import os

from agent.utils import load_txt_to_str

command_config = configparser.ConfigParser()
command_config.read('command/command.ini', encoding='utf-8-sig')


# 指令执行记号，是一个全局变量，用于作用到整个系统
class CommandFlags:
    def __init__(self):
        self.reset()

    def reset(self):
        self.not_command = True
        self.wrong_command = False
        self.history = False
        self.show_temp_history = False
        self.prompt = False
        self.show_prompt = False
        self.event = False
        self.show_event = False
        self.retry = False
        self.help = False
        self.exit = False
        self.user_name = ''
        self.ai_name = ''


command_flags = CommandFlags()


class Pool:

    def __init__(self):
        self.command_start = command_config.get('START', 'command_start')

    def execute(self, command: str, var_dict: dict):
        """
        :param command: 指令字符串
        :param var_dict: 携带的变量字典，方便指令处理时获取信息
        :return: 指令执行结果，用于与调用方通信
        """
        if command[0] != self.command_start:
            return
        command_flags.not_command = False
        if len(command) == 1:
            print("（您输入了空指令，输入'" + self.command_start + "help'来获取帮助。'）")
            command_flags.wrong_command = True
            return 'wrong command'
        if command[1:] not in command_config['LIST'].values():
            print("（指令不存在，输入'" + self.command_start + "help'来获取帮助。'）")
            command_flags.wrong_command = True
            return 'wrong command'

        command_flags.ai_name = var_dict['info'].ai_name
        # command_flags.user_name = var_dict['info'].user_name
        command_list = command_config['LIST']
        # 以下是指令处理部分，该部分可以自定义指令
        # 打开对话历史文件
        command = command[1:]
        # 打开文件操作暂时只支持Windows系统
        if command == command_list['history']:
            try:
                path = os.path.abspath(var_dict['info'].history_path)
                os.startfile(path)
                command_flags.history = True
                return 'history'
            except AttributeError:
                pass
        elif command == command_list['prompt']:
            try:
                path = os.path.abspath(var_dict['info'].prompt_path)
                os.startfile(path)
                command_flags.prompt = True
                return 'prompt'
            except AttributeError:
                pass
        elif command == command_list['event']:
            try:
                path = os.path.abspath(var_dict['info'].event_path)
                os.startfile(path)
                command_flags.event = True
                return 'event'
            except AttributeError:
                pass
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
        elif command == command_list['show_temp_history']:
            # 展示临时窗口历史对话
            command_flags.show_temp_history = True
            return 'show_temp_history'
        elif command == command_list['show_prompt']:
            # 展示临时窗口历史对话
            command_flags.show_prompt = True
            return 'show_prompt'
        elif command == command_list['show_event']:
            # 展示临时窗口历史对话
            command_flags.show_event = True
            return 'show_event'
        else:
            print("（指令不存在，输入'" + self.command_start + "help'来获取帮助。'）")
            command_flags.wrong_command = True
        return 'no process'
