import copy
import os
import re

import textdistance
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from agent.abstract_agent import AbstractAgent
from agent.audio import AudioModule
from agent.utils import init_knowledge_vector_store, load_txt_to_lst, CharacterInfo, delete_last_line, \
    load_last_n_lines, append_to_str_file
from agent.llm import Gpt3_5LLM, ChatGLMLLM, Gpt3_5freeLLM
from world_manager import CharacterInfo
from command import Pool, command_flags

embed_model_path = 'text2vec/GanymedeNil_text2vec-large-chinese'
# embed_model_path = 'text2vec/shibing624_text2vec_base_chinese'
# embed_model_path = 'GanymedeNil/text2vec-large-chinese'
device = 'cuda'
DEBUG_MODE = True


def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def collect_context(text_lst):
    return "\n".join([text for text in text_lst])


class VectorStore:

    def __init__(self, embeddings, path, chunk_size=20, top_k=6):
        self.top_k = top_k
        self.path = path
        self.core, _ = init_knowledge_vector_store(embeddings=embeddings, filepath=self.path)
        self.chunk_size = chunk_size
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector

    def similarity_search_with_score(self, query):
        if self.core is not None:
            self.core.chunk_size = self.chunk_size
            return self.core.similarity_search_with_score(query, self.top_k)
        else:
            return []

    def get_path(self):
        return self.path


def get_related_text_lst(query, vs, lst):
    related_text_with_score = vs.similarity_search_with_score(query)
    lst.extend(get_docs_with_score(related_text_with_score))


def mem_to_lst(mem):
    lst = []
    for i in range(len(mem)):
        lst.append(mem[i].page_content)
    return lst


class MainAgent(AbstractAgent):

    def __init__(self,
                 world_name,
                 ai_name,
                 user_name='user',
                 model_name='gpt3_5',
                 lock_memory=False,
                 history_window=3,
                 window_decrease_size=400,
                 classifier_enabled=False,
                 max_history_size=1100,
                 context_chunk_size=20,
                 temperature=0.01,
                 streaming=True,
                 memory_search_top_k=3,
                 embedding_model_device=device,
                 speak_rate='快'):
        """

        :param world_name: 世界名称，不同世界对应不同人设。
        :param model_name: 模型名称，可用：chatglm-6b-int4、gpt3_5。
        :param lock_memory: 锁定对话记忆，为False时每次对话都要读取记忆文件，速度较慢，但每次对话增加新的对话到记忆文件中；
                            为True仅第一次加载时读取记忆文件，速度较快，但不增加新的记忆。
        :param classifier_enabled: 话题、情绪分类器是否可用。
        :param max_history_size: 在内存里的最大对话历史窗口token大小。
        :param context_chunk_size: 上下文chunk数量，越大则能载入更多记忆内容。
        :param streaming: 流式输出。
        :param memory_search_top_k: 记忆和提问的相关性最高前k个提取到提问上下文中。
        :param embedding_model_device: embedding模型的device（可选参数为 'cpu'、'cuda'）。
        :param speak_rate: 阅读回答的速度。
        """
        super().__init__()
        self.world_name = world_name
        self.ai_name = ai_name
        self.info = CharacterInfo(self.world_name, self.ai_name)
        self.user_name = user_name
        self.memory_search_top_k = memory_search_top_k
        self.context_chunk_size = context_chunk_size
        self.lock_memory = lock_memory
        self.history_window = history_window
        self.window_decrease_size = window_decrease_size
        self.query = ''  # 提问的临时存储，用于重试提问
        if speak_rate == '快':
            rate = 200
        elif speak_rate == '中':
            rate = 150
        elif speak_rate == '慢':
            rate = 100
        else:
            rate = 150

        self.streaming = streaming
        embedding_model_path = embed_model_path
        embedding_device = embedding_model_device
        # self.streaming = streaming
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path,
                                                model_kwargs={'device': embedding_device})

        self.identity_top_k = 2
        self.history_top_k = 9
        self.event_top_k = 3
        self.semantic_similarity_threshold = 20.0  # 语义相似度分数阈值（相似度太小的会被去掉）
        self.word_similarity_threshold = 0.6  # 字词相似度分数阈值（相似度太大的会被去掉）
        self.identity_vs = VectorStore(self.embeddings, self.info.identity_path, chunk_size=30,
                                       top_k=self.identity_top_k)
        if self.lock_memory:
            self.history_vs = VectorStore(self.embeddings, self.info.history_path, chunk_size=30,
                                          top_k=self.history_top_k)
        self.event_vs = VectorStore(self.embeddings, self.info.event_path, chunk_size=30,
                                    top_k=self.event_top_k)
        print("【---记忆模块加载完成---】")
        # ---model
        self.model_name = model_name
        if self.model_name == 'chatglm-6b-int4':
            self.path = 'THUDM/chatglm-6b-int4'
            # self.path = 'chatglm-6b-int4'
            self.llm = ChatGLMLLM(temperature=temperature)
            self.llm.load_model(self.path)
        elif self.model_name == 'gpt3_5':
            self.llm = Gpt3_5LLM(temperature=temperature)
        elif model_name == 'gpt3_5free':
            self.llm = Gpt3_5freeLLM(temperature=temperature)
        else:
            raise AttributeError("模型选择参数出错！传入的参数为", self.model_name)
        # 历史对话列表
        self.history = []
        # 初始化提示语
        self.basic_history = load_txt_to_lst(self.info.prompt_path)
        # 加载短期对话历史
        # self.llm.load_history(self.basic_history)
        self.load_history(self.basic_history)
        # 窗口控制
        # self.llm.set_max_history_size(max_history_size)
        self.basic_token_len = 0
        self.total_token_size = 0
        self.max_history_size = max_history_size
        print("【---对话模型加载完成---】")
        # ---voice
        self.voice_module = AudioModule(sound_library='local', rate=rate)
        print("【---声音模块加载完成---】")
        # ---
        # ---话题分类器
        # self.classifier_enabled = classifier_enabled
        # if self.classifier_enabled:
        #     # self.classifier = Classifier()
        #     print("【---话题分类器加载完成---】")

    def chat(self, query):
        # ------指令部分
        self.check_command()
        var_dict = {'info': self.info}
        Pool().execute(query, var_dict)
        if command_flags.exit:
            # 执行退出指令
            return 'ai_chat_with_memory sys:exit'
        # 若query是一个指令，则处理过后退出，不进行对话
        if not command_flags.not_command and not command_flags.retry:
            self.check_show_command()
            return 'ai_chat_with_memory sys:执行了指令'

        if command_flags.retry:
            # 执行重试指令
            if self.query == '':
                print("当前没有提问，请输入提问。")
                return 'ai_chat_with_memory sys:当前没有提问，无法重试提问。'
            # 从临时存储中取出提问
            query = self.query
            if not self.lock_memory:
                # 删除历史文件最后一行
                delete_last_line(self.info.history_path)
                # 重新加载临时历史对话
                self.load_history(self.basic_history)
        # ------

        # ------文本检索部分
        # 检索记忆（实体、对话、事件）
        entity_lst, dialog_lst, event_lst = self.get_related_text(query)
        # 嵌入提示词
        entity_text = collect_context(entity_lst)
        dialog_text = collect_context(dialog_lst)
        event_text = collect_context(event_lst)
        context_len = self.embedding_context(entity_text, dialog_text, event_text)
        # ------

        # 文本中加入提问者身份
        if self.user_name != '':
            q = self.user_name + "说：" + query + '\n' + self.ai_name + '说：'
        else:
            q = query + '\n' + self.ai_name + '说：'

        # ---与大模型通信
        ans = self.llm.chat(q, self.history)
        # ---

        self.history.append((query, ans))
        self.total_token_size += (len(ans) + len(query))

        print("Token size:", self.total_token_size + context_len)

        # 恢复最开头的提示词
        self.history[0] = self.basic_history[0]

        if not self.lock_memory:
            # 保存历史到文件中
            append_str = q.replace('\n', ' ') + ans + '\n'
            append_to_str_file(self.info.history_path, append_str)
        # 窗口控制
        self.history_window_control(context_len)

        # if self.classifier_enabled:
        #     topic_tag = self.classifier.do(ans)
        #     print(self.ai_name, ":{}\n".format(ans), "话题：", topic_tag)
        # else:
        print(self.ai_name, ":{}\n".format(ans))
        # self.voice_module.say(ans)
        # 临时存储当前提问
        self.query = query
        return ans

    def history_window_control(self, context_len):
        if self.total_token_size + context_len >= self.max_history_size:
            while self.total_token_size + context_len > (self.max_history_size - self.window_decrease_size):
                try:
                    self.total_token_size -= (len(self.history[1][0]) + len(self.history[1][1]))
                    self.history.pop(1)
                except IndexError:
                    # print("窗口不能再缩小了")
                    break
            if DEBUG_MODE:
                print("窗口缩小， 历史对话：")
                print(self.history)

    def load_history(self, basic_history):
        self.basic_history = basic_history
        self.basic_token_len = len(self.basic_history[0][0]) + len(self.basic_history[0][1])
        self.total_token_size = 0
        if os.path.exists(self.info.history_path) and os.path.getsize(self.info.history_path) == 0:
            # 历史记录为空
            self.history = copy.deepcopy(self.basic_history)
        else:
            self.history = copy.deepcopy(self.basic_history)
            # 加载历史记录最后几行
            history_lst = load_last_n_lines(self.info.history_path, self.history_window)
            pattern = r'(.+?) ' + self.info.ai_name + '说：(.+?)(?=$)'

            for dialog in history_lst:
                matches = re.findall(pattern, dialog)
                self.history.append(matches[0])

    def embedding_context(self, entity, dialog, event):

        entity = entity.replace("{{{AI_NAME}}}", self.ai_name)
        dialog = dialog.replace("{{{AI_NAME}}}", self.ai_name)
        event = event.replace("{{{AI_NAME}}}", self.ai_name)

        context = self.history[0][0]
        context = context.replace("{{{ENTITY}}}", entity)
        context = context.replace("{{{DIALOG}}}", dialog)
        context = context.replace("{{{EVENT}}}", event)
        context = context.replace("{{{AI_NAME}}}", self.ai_name)
        context = context.replace("{{{USER_NAME}}}", self.user_name)

        first_ans = self.history[0][1].replace("{{{AI_NAME}}}", self.ai_name)

        context_len = len(entity) + len(dialog) + len(event)
        if DEBUG_MODE:
            print("context长度:", context_len)
            print("提示词总长度:", len(context))

        self.history[0] = (context, first_ans)
        print(self.history[0])

        if DEBUG_MODE:
            print("实体记忆：")
            print(entity)
            print("对话记忆：")
            print(dialog)
            print("事件记忆：")
            print(event)
        return context_len

    def get_related_text(self, query):
        # 实体记忆
        entity_mem = []
        get_related_text_lst(query, self.identity_vs, entity_mem)
        entity_lst = mem_to_lst(entity_mem)
        # 字词高相似度去重
        entity_lst = self.high_similarity_text_filter(entity_lst)

        # 对话记忆
        dialog_mem = []
        if self.lock_memory:
            get_related_text_lst(query, self.history_vs, dialog_mem)
        else:
            get_related_text_lst(query,
                                 VectorStore(self.embeddings, self.info.history_path,
                                             chunk_size=10,
                                             top_k=self.history_top_k),
                                 dialog_mem)
        dialog_lst = mem_to_lst(dialog_mem)
        dialog_lst = self.high_similarity_text_filter(dialog_lst)

        # 事件记忆
        event_mem = []
        get_related_text_lst(query, self.event_vs, event_mem)
        event_lst = mem_to_lst(event_mem)
        event_lst = self.high_similarity_text_filter(event_lst)

        return entity_lst, dialog_lst, event_lst

    def high_similarity_text_filter(self, text_lst):
        # 字词相似度比对，算法复杂度o(n^2)
        remaining_strings = list(text_lst)  # 创建一个副本以避免在迭代时修改原始列表
        for i in range(len(text_lst)):
            for j in range(len(text_lst)):
                if i != j:
                    sim_score = textdistance.jaccard(text_lst[i], text_lst[j])
                    if sim_score > self.word_similarity_threshold:
                        # 如果两个字符串的相似度超过阈值，则删除较短的字符串（较短的字符串信息含量大概率较少）
                        del_str = text_lst[i] if len(text_lst[i]) < len(text_lst[j]) else text_lst[j]
                        if del_str in remaining_strings:
                            remaining_strings.remove(del_str)
        return remaining_strings

    def check_command(self):
        if command_flags.ai_name != self.ai_name:
            return
        if command_flags.history:
            if self.lock_memory:
                # 历史对话被打开过，重新加载历史对话（仅当lock_memory为True时重新加载）
                self.history_vs = VectorStore(self.embeddings, self.info.history_path, chunk_size=1,
                                              top_k=self.history_top_k)
            # 重新加载临时历史对话
            self.load_history(self.basic_history)
        elif command_flags.prompt:
            # 提示词被打开过，重新加载提示词和历史对话
            self.basic_history = load_txt_to_lst(self.info.prompt_path)
            self.load_history(self.basic_history)
        command_flags.reset()

    def check_show_command(self):
        if command_flags.show_temp_history:
            # 展示当前临时历史窗口
            window = self.history
            for dialog in window[1:]:
                print(dialog[0])
                print(self.ai_name + '说：' + dialog[1])
        elif command_flags.show_prompt:
            # 展示当前提示词
            print("提示词：")
            print(self.basic_history[0][0], end='')
            print(self.basic_history[0][1])
