from command import command_cleanup_task, Pool, command_flags, execute_command
from tools.utils import delete_last_line


class Sandbox:

    def __init__(self, world_name):
        self.world_name = world_name
        self.chat_str = ''
        self.default_user_name = ''

    def chat_with_one_agent(self, agent):
        print("---初始化完成，对话开始---")
        print("'输入/help'可获取指令列表")
        self.default_user_name = agent.user_name
        while True:
            self.chat_str = ''
            while self.chat_str == '':
                self.chat_str = input((self.default_user_name + "：") if self.default_user_name != '' else 'user：')
            back_msg = self.check_command(self.chat_str, agent)
            if back_msg == 'ai_chat_with_memory sys:exit':
                return
            if back_msg == 'chat':
                agent.chat(self.chat_str)

    def check_command(self, query, agent):
        # ------指令部分
        # 指令收尾工作
        if command_flags.continue_talk:
            agent.set_user_name(self.default_user_name)
        command_cleanup_task(agent)
        # 检查是否为指令
        Pool().check(query, agent.ai_name)
        if not command_flags.not_command:
            sys_mes = execute_command(agent)
            # 执行重试指令
            if command_flags.retry:
                if agent.query == '':
                    print("当前没有提问，请输入提问。")
                    return 'ai_chat_with_memory sys:当前没有提问，无法重试提问。'
                # 从临时存储中取出提问
                self.chat_str = agent.get_tmp_query()
                if not agent.lock_memory:
                    # 删除历史文件最后一行
                    delete_last_line(agent.info.history_path)
                    # 重新加载临时历史对话
                    agent.load_history(agent.basic_history)
                agent.step -= 1
            # 执行
            elif command_flags.continue_talk:
                last_ans = agent.last_ans
                self.chat_str = last_ans
                if self.chat_str == '':
                    # 加载最后的历史对话
                    splitter = agent.ai_name + '说'
                    ans_str = agent.history[-1][1]
                    self.chat_str = ans_str[ans_str.find(splitter) + len(splitter):].replace('\n', '')
                agent.set_user_name(agent.ai_name)
            else:
                return 'ai_chat_with_memory sys:执行了指令。'
        elif command_flags.wrong_command:
            return 'ai_chat_with_memory sys:错误指令。'
        return 'chat'
        # ------
