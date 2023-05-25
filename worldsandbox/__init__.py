from command import command_cleanup_task, Pool, command_flags, execute_command
from tools.utils import delete_last_line


class Sandbox:

    def __init__(self, world_name):
        self.world_name = world_name
        self.chat_str = ''

    def chat_with_one_agent(self, agent):
        print("---初始化完成，对话开始---")
        print("'输入/help'可获取指令列表")
        while True:
            self.chat_str = ''
            while self.chat_str == '':
                self.chat_str = input((agent.user_name + "：") if agent.user_name != '' else 'user：')
            back_msg = self.check_command(self.chat_str, agent)
            if back_msg == 'ai_chat_with_memory sys:exit':
                return
            if back_msg == 'no command':
                agent.chat(self.chat_str)

    def check_command(self, query, agent):
        # ------指令部分
        # 指令收尾工作
        command_cleanup_task(agent)
        # 检查是否为指令
        Pool().check(query, agent.ai_name)
        if not command_flags.not_command:
            sys_mes = execute_command(agent)
            # 执行了除重试指令以外的指令，不进行对话
            if sys_mes != '':
                return sys_mes
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
                return 'no command'
        else:
            return 'no command'
        # ------
