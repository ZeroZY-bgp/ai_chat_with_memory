import importlib
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from agent import MainAgent
from command import command_cleanup_task, Pool, command_flags, execute_command
from tools.utils import delete_last_line


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_llm(llm_class_name, temperature, device):
    llm_class = get_class("agent.llm", llm_class_name)
    llm_instance = llm_class(temperature)
    if hasattr(llm_instance, 'load_model'):
        llm_instance.load_model()
    if hasattr(llm_instance, 'set_device'):
        llm_instance.set_device(device)
    return llm_instance


def create_embedding_model(embed_model_path, embed_device):
    return HuggingFaceEmbeddings(model_name=embed_model_path,
                                 model_kwargs={'device': embed_device})


class Sandbox:

    def __init__(self, world_name):
        self.world_name = world_name
        self.chat_str = ''
        self.default_user_name = ''
        self.one_agent = True
        self.use_embed_model = True
        self.language_model = None
        self.embedding_model = None
        self.multi_agent_chat_strategy = 'round'

    def set_models(self, config):
        self.language_model = create_llm(config['MODEL']['name'],
                                         config.getfloat('MODEL', 'temperature'),
                                         config['MODEL']['model_device'])
        self.use_embed_model = config.getboolean('MODEL', 'use_embed_model')
        if self.use_embed_model:
            self.embedding_model = create_embedding_model(config['MODEL']['embed_model'],
                                                          config['MODEL']['embed_model_device'])

    def init_agent(self, config, world_name, ai_name):
        return MainAgent(world_name, ai_name, self.language_model, self.embedding_model, config)

    def chat_with_one_agent(self, config):
        self.one_agent = True
        world_name = config['WORLD']['name']
        ai_name = config['AI']['name']
        self.set_models(config)
        agent = self.init_agent(config, world_name, ai_name)
        self.default_user_name = agent.user_name
        print("---初始化完成，对话开始---")
        print("'输入/help'可获取指令列表")
        while True:
            self.chat_str = ''
            while self.chat_str == '':
                self.chat_str = input((self.default_user_name + "：") if self.default_user_name != '' else 'user：')
            back_msg = self.check_command(self.chat_str, agent)
            if back_msg == 'exit':
                return
            if back_msg == 'chat':
                agent.chat(self.chat_str)

    def chat_with_multi_agent(self, agent_str_lst: List[str], config):
        if len(agent_str_lst) == 0:
            print("空ai列表，请检查参数")
            return
        agent_lst = {}
        for name in agent_str_lst:
            agent_lst[name] = self.init_agent(config, config['WORLD']['name'], name)
        if self.multi_agent_chat_strategy == 'round':
            start_prompt = ''
            self.chat_str = start_prompt
            pre_agent_name = agent_lst[0].ai_name
            for agent in agent_lst:
                agent.user_name = pre_agent_name
                self.chat_str = agent.chat(self.chat_str)
                pre_agent_name = agent.ai_name
                input("按任意键继续")

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
                    # 无最后的回答缓存，加载最后的历史对话
                    splitter = agent.ai_name + '说'
                    ans_str = agent.history[-1][1]
                    self.chat_str = ans_str[ans_str.find(splitter) + len(splitter):].replace('\n', '')
                agent.set_user_name(agent.ai_name)
            elif command_flags.exit:
                return 'exit'
            else:
                return 'ai_chat_with_memory sys:执行了指令。'
        elif command_flags.wrong_command:
            return 'ai_chat_with_memory sys:错误指令。'
        return 'chat'
        # ------
