import copy
import os
import random

from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from tools.audio import AudioModule
from tools.text_splitter import AnswerTextSplitter, high_word_similarity_text_filter, \
    low_semantic_similarity_text_filter
from tools.utils import load_txt_to_lst, delete_last_line, load_last_n_lines, append_to_str_file, VectorStore
from agent.llm import Gpt3_5LLM, ChatGLMLLM, Gpt3_5Useless, Gpt3Deepai
from world_manager import CharacterInfo
from command import Pool, command_flags, execute_command, command_cleanup_task

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


def get_related_text_lst(query, vs, lst):
    related_text_with_score = vs.similarity_search_with_score(query)
    lst.extend(get_docs_with_score(related_text_with_score))


def mem_to_lst(mem):
    lst = []
    for i in range(len(mem)):
        lst.append(mem[i].page_content)
    return lst


class MainAgent:

    def __init__(self,
                 world_name,
                 ai_name,
                 user_name='user',
                 model_name='gpt3_5',
                 lock_memory=False,
                 history_window=3,
                 window_decrease_size=400,
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
        :param max_history_size: 在内存里的最大对话历史窗口token大小。
        :param context_chunk_size: 上下文chunk数量，越大则能载入更多记忆内容。
        :param streaming: 流式输出。
        :param memory_search_top_k: 记忆和提问的相关性最高前k个提取到提问上下文中。
        :param embedding_model_device: embedding模型的device（可选参数为 'cpu'、'cuda'）。
        :param speak_rate: 阅读回答的速度。
        """
        self.world_name = world_name
        self.ai_name = ai_name
        self.info = CharacterInfo(self.world_name, self.ai_name)
        self.user_name = user_name
        self.memory_search_top_k = memory_search_top_k
        self.context_chunk_size = context_chunk_size
        self.lock_memory = lock_memory
        self.history_window = history_window
        self.window_decrease_size = window_decrease_size
        # ---暂存区
        self.query = ''  # 提问的临时存储，用于重试提问
        self.entity_text = ''
        self.dialog_text = ''
        self.event_text = ''
        self.last_ans = ''

        # ---
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

        # ------高级开发参数
        self.entity_top_k = 2
        self.history_top_k = 8
        self.event_top_k = 1
        self.semantic_similarity_threshold = 100.0  # 语义相似度分数阈值（相似度太小的会被去掉）
        self.word_similarity_threshold = 0.6  # 字词相似度分数阈值（相似度太大的会被去掉）
        self.update_history_vs_per_step = 6  # 不锁定记忆的情况下，更新记忆vector store的频率（数值越小更新频率越大）
        self.similarity_comparison_context_window = 3  # 用于相似度比较的对话上文窗口
        self.answer_extract_enabled = True  # 对话记忆仅提取ai回答
        self.fragment_answer = False  # 将回答打碎

        # ------
        self.step = 1

        # vector store
        self.entity_textsplitter = CharacterTextSplitter(separator="\n")
        self.entity_vs = VectorStore(self.embeddings, self.info.entity_path, chunk_size=30,
                                     top_k=self.entity_top_k, textsplitter=self.entity_textsplitter)
        self.history_textsplitter = CharacterTextSplitter(separator="\n")
        self.history_vs = VectorStore(self.embeddings, self.info.history_path, chunk_size=30,
                                      top_k=self.history_top_k, textsplitter=self.history_textsplitter)
        self.event_textsplitter = CharacterTextSplitter(separator="\n")
        self.event_vs = VectorStore(self.embeddings, self.info.event_path, chunk_size=30,
                                    top_k=self.event_top_k, textsplitter=self.event_textsplitter)
        print("【---记忆模块加载完成---】")
        # ---model（此处加入自定义包装的模型）
        self.model_name = model_name
        if self.model_name == 'chatglm-6b-int4':
            self.path = 'THUDM/chatglm-6b-int4'
            # self.path = 'chatglm-6b-int4'
            self.llm = ChatGLMLLM(temperature=temperature)
            self.llm.load_model(self.path)
        elif self.model_name == 'gpt3_5':
            self.llm = Gpt3_5LLM(temperature=temperature)
        elif model_name == 'gpt3_5useless':
            self.llm = Gpt3_5Useless(temperature=temperature)
        elif model_name == 'gpt3deepai':
            self.llm = Gpt3Deepai(temperature=temperature)
        else:
            raise AttributeError("模型选择参数出错！传入的参数为", self.model_name)
        # 历史对话列表
        self.history = []
        # 初始化提示语
        self.basic_history = load_txt_to_lst(self.info.prompt_path)
        self.cur_prompt = self.basic_history[0][0]
        # 加载短期对话历史
        self.load_history(self.basic_history)
        # 窗口控制
        self.basic_token_len = 0
        self.total_token_size = 0
        self.max_history_size = max_history_size
        print("【---对话模型加载完成---】")
        # ---voice
        self.voice_module = AudioModule(sound_library='local', rate=rate)
        print("【---声音模块加载完成---】")
        # ---

    def chat(self, query):
        # ------指令部分
        # 指令收尾工作
        command_cleanup_task(self)
        # 检查是否为指令
        Pool().check(query, self.ai_name)
        if not command_flags.not_command:
            # sys_mes = self.execute_command(query)
            sys_mes = execute_command(self)
            # 执行了除重试指令以外的指令，不进行对话
            if sys_mes != '':
                return sys_mes
            # 执行重试指令
            if command_flags.retry:
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
                self.step -= 1
        # ------

        # 文本中加入提问者身份
        q_start = self.user_name + "说：" if self.user_name != '' else ''
        # ------检索记忆（实体、对话、事件）
        # 获取上文窗口
        entity_lst, dialog_lst, event_lst = self.get_related_text(self.get_context_window(q_start + query))
        # 嵌入提示词
        self.entity_text = collect_context(entity_lst)
        self.dialog_text = collect_context(dialog_lst)
        self.event_text = collect_context(event_lst)
        context_len = self.embedding_context(self.entity_text, self.dialog_text, self.event_text)
        # ------

        # ---与大模型通信
        # res = openai.Moderation.create(
        #     input=q_start
        # )
        # print(res["results"][0])
        ans = self.llm.chat(q_start + query + '\n' + self.ai_name + '说：', self.history)
        # res = openai.Moderation.create(
        #     input=ans
        # )
        # print(res["results"][0])
        # ---

        # ---处理对话历史
        self.cur_prompt = self.history[0][0]
        self.history.append((q_start + query, self.ai_name + '说：' + ans))
        self.total_token_size += (len(ans) + len(query))

        print("Token size:", self.total_token_size + context_len)

        # 恢复最开头的提示词
        self.history[0] = self.basic_history[0]

        if not self.lock_memory:
            # 保存历史到文件中
            append_str = q_start + query + ' ' + self.ai_name + '说：' + ans + '\n'
            append_to_str_file(self.info.history_path, append_str)
        # 窗口控制
        self.history_window_control(context_len)
        # ---

        print(self.ai_name, ":{}\n".format(ans))
        # self.voice_module.say(ans)
        # 临时存储当前提问
        self.query = query
        return ans

    def get_context_window(self, query):
        lines = load_last_n_lines(self.info.history_path, self.similarity_comparison_context_window - 1)
        comparison_string = ' '.join(line for line in lines)
        comparison_string += query
        return comparison_string

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
            splitter = self.info.ai_name + '说'
            for dialog in history_lst:
                first_index = dialog.find(splitter)
                second_index = dialog.find(splitter, first_index + len(splitter))
                if second_index != -1:
                    # 两次AI回复，说明是继续回答
                    tuple_result = (dialog[:second_index], dialog[second_index:])
                else:
                    tuple_result = (dialog[:first_index], dialog[first_index:])
                self.history.append(tuple_result)

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
        get_related_text_lst(query, self.entity_vs, entity_mem)
        # 字词高相似度去重
        entity_mem = high_word_similarity_text_filter(self, entity_mem)
        # 语义低相似度去重
        entity_mem = low_semantic_similarity_text_filter(self, entity_mem)
        entity_lst = mem_to_lst(entity_mem)

        # 对话记忆
        dialog_mem = []
        if not self.lock_memory and self.step >= self.update_history_vs_per_step:
            self.history_vs = VectorStore(self.embeddings, self.info.history_path,
                                          chunk_size=10,
                                          top_k=self.history_top_k,
                                          textsplitter=self.history_textsplitter)
            self.step = 1
            if DEBUG_MODE:
                print("History vector store updated.")

        self.step += 1

        get_related_text_lst(query, self.history_vs, dialog_mem)
        if self.answer_extract_enabled:
            self.answer_extract(dialog_mem)
            if self.fragment_answer:
                tsp = AnswerTextSplitter()
                docs = tsp.split_documents(dialog_mem)
                db = FAISS.from_documents(docs, self.embeddings)
                dialog_mem = db.similarity_search_with_score(query, len(docs))
                dialog_mem = get_docs_with_score(dialog_mem)

        dialog_mem = high_word_similarity_text_filter(self, dialog_mem)
        dialog_mem = low_semantic_similarity_text_filter(self, dialog_mem)
        dialog_lst = mem_to_lst(dialog_mem)

        # 事件记忆
        event_mem = []
        get_related_text_lst(query, self.event_vs, event_mem)
        event_mem = high_word_similarity_text_filter(self, event_mem)
        event_mem = low_semantic_similarity_text_filter(self, event_mem)
        event_lst = mem_to_lst(event_mem)

        # 随机打乱列表
        random.shuffle(entity_lst)
        random.shuffle(dialog_lst)
        random.shuffle(event_lst)

        return entity_lst, dialog_lst, event_lst

    def answer_extract(self, mem, has_ai_name=True):
        # 提取对话，仅有ai的回答
        splitter = self.ai_name + '说：'
        for dialog in mem:
            parts = dialog.page_content.split(splitter)
            dialog.page_content = splitter + parts[-1] if has_ai_name else parts[-1]

