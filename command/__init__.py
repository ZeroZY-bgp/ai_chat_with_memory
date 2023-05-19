import configparser
import os
import re

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
        self.open = False

        self.history = False
        self.prompt = False
        self.event = False
        self.entity = False
        self.show_temp_history = False
        self.show_prompt = False
        self.show_context = False

        self.dialog_to_event = False
        self.retry = False
        self.help = False
        self.exit = False
        self.user_name = ''
        self.ai_name = ''


command_flags = CommandFlags()


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
        elif command == command_list['dialog_to_event']:
            command_flags.dialog_to_event = True
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
