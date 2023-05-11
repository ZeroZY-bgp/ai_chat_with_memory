import re
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from agent.abstract_agent import AbstractAgent
from agent.audio import AudioModule
from agent.utils import init_knowledge_vector_store
from agent.llm import Gpt3_5LLM, ChatGLMLLM, Gpt3_5freeLLM

embed_model_path = 'text2vec/GanymedeNil_text2vec-large-chinese'
# embed_model_path = 'GanymedeNil/text2vec-large-chinese'
device = 'cpu'

def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def collect_context(text_lst):
    return "\n".join([text.page_content for text in text_lst])


def generate_prompt(context: str,
                    query: str,
                    prompt_template) -> str:
    prompt = prompt_template.replace("{query}", query).replace("{context}", context)
    return prompt


def get_txt_addr(text, world_name):
    pattern = r'agent/memory/' + world_name + '/[^/]+/[^/]+\.txt'  # 匹配任意倒数第二级目录名称
    file_list = re.findall(pattern, text)
    return file_list


def load_txt_to_lst(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return eval(text)


class VectorStore:

    def __init__(self, embeddings, path, chunk_size=20, top_k=6):
        self.top_k = top_k
        self.path = path
        self.vs_path, _ = init_knowledge_vector_store(embeddings=embeddings, filepath=self.path)
        if self.vs_path is None:
            self.core = None
        else:
            self.core = FAISS.load_local(self.vs_path, embeddings)
        # self.core.chunk_size = chunk_size
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector

    def similarity_search_with_score(self, query):
        if self.core is not None:
            return self.core.similarity_search_with_score(query, self.top_k)
        else:
            return []

    def get_path(self):
        return self.path


def get_related_text_lst(query, vs, lst):
    related_text_with_score = vs.similarity_search_with_score(query)
    lst.extend(get_docs_with_score(related_text_with_score))


class MainAgent(AbstractAgent):

    def __init__(self,
                 world_name,
                 ai_name,
                 user_name='user',
                 model_name='gpt3_5',
                 lock_memory=False,
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
        self.user_name = user_name
        self.memory_search_top_k = memory_search_top_k
        self.context_chunk_size = context_chunk_size
        self.lock_memory = lock_memory
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
        self.index_path = 'agent/memory/' + self.world_name + '/index.txt'
        embedding_device = embedding_model_device
        # self.streaming = streaming
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path,
                                                model_kwargs={'device': embedding_device})
        self.history_file = 'agent/memory/' + self.world_name + '/local/' + self.ai_name + '/history.txt'
        self.identity_file = 'agent/memory/' + self.world_name + '/global/all.txt'
        self.event_file = 'agent/memory/' + self.world_name + '/local/' + self.ai_name + '/event.txt'
        self.file_lst = []
        self.identity_vs = VectorStore(self.embeddings, self.identity_file, chunk_size=1,
                                       top_k=2)
        if self.lock_memory:
            self.history_vs = VectorStore(self.embeddings, self.history_file, chunk_size=1,
                                          top_k=9)
        self.event_vs = VectorStore(self.embeddings, self.event_file, chunk_size=1,
                                    top_k=6)
        print("【---记忆模块加载完成---】")
        # ---model
        self.model_name = model_name
        if self.model_name == 'chatglm-6b-int4':
            # self.path = 'THUDM/chatglm-6b-int4'
            self.path = 'chatglm-6b-int4'
            self.llm = ChatGLMLLM(self.ai_name, user_name=self.user_name, world_name=self.world_name,
                                  lock_memory=self.lock_memory, temperature=temperature)
            self.llm.load_model(self.path)
        elif self.model_name == 'gpt3_5':
            self.llm = Gpt3_5LLM(ai_name=self.ai_name, user_name=self.user_name, world_name=self.world_name,
                                 lock_memory=self.lock_memory, temperature=temperature)
        elif model_name == 'gpt3_5free':
            self.llm = Gpt3_5freeLLM(ai_name=self.ai_name, user_name=self.user_name, world_name=self.world_name,
                                     lock_memory=self.lock_memory, temperature=temperature)
        else:
            raise AttributeError("模型选择参数出错！传入的参数为", self.model_name)
        # 初始化提示语
        basic_prompt_path = 'agent/memory/' + self.world_name + '/prompts/' + self.ai_name + '.txt'
        basic_history = load_txt_to_lst(basic_prompt_path)
        # 加载短期对话历史
        self.llm.load_history(basic_history)
        # 窗口控制
        self.llm.set_max_history_size(max_history_size)
        # ---
        print("【---对话模型加载完成---】")
        # ---voice
        self.voice_module = AudioModule(sound_library='local', rate=rate)
        print("【---声音模块加载完成---】")
        # ---
        # ---话题分类器
        self.classifier_enabled = classifier_enabled
        if self.classifier_enabled:
            # self.classifier = Classifier()
            print("【---话题分类器加载完成---】")

    def chat(self, query):
        # 相似文本列表
        related_text_lst = []
        # 直接加载记忆模块
        # 身份，世界观记忆
        get_related_text_lst(query, self.identity_vs, related_text_lst)
        # 对话记忆
        if self.lock_memory:
            get_related_text_lst(query, self.history_vs, related_text_lst)
        else:
            get_related_text_lst(query,
                                 VectorStore(self.embeddings, self.history_file,
                                             chunk_size=10,
                                             top_k=6),
                                 related_text_lst)
        # 事件记忆
        get_related_text_lst(query,
                             VectorStore(self.embeddings, self.event_file,
                                         chunk_size=10,
                                         top_k=3),
                             related_text_lst)

        context = collect_context(related_text_lst)
        response = self.llm.chat(self.user_name + "说：" + query + '\n' + self.ai_name + '说：', context)

        if self.classifier_enabled:
            topic_tag = self.classifier.do(response)
            print(self.ai_name, ":{}\n".format(response), "话题：", topic_tag)
        else:
            print(self.ai_name, ":{}\n".format(response))
        # self.voice_module.say(response)
