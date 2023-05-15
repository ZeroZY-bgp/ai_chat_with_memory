import datetime
import os
import re
from typing import List

from langchain import FAISS
from langchain.document_loaders import UnstructuredFileLoader

from agent.chinese_text_splitter import ChineseTextSplitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")


class CharacterInfo:
    def __init__(self, world_name, ai_name):
        self.world_name = world_name
        self.ai_name = ai_name
        self.folder_path = 'agent/memory/' + self.world_name + '/' + self.ai_name
        self.prompt_path = self.folder_path + '/prompt' + self.ai_name + '.txt'
        self.history_path = self.folder_path + '/history' + self.ai_name + '.txt'
        self.identity_path = self.folder_path + '/identity' + self.ai_name + '.txt'
        self.event_path = self.folder_path + '/event' + self.ai_name + '.txt'


def get_tag(string):
    topic_tag = ''
    emotion_tag = ''
    # 提取##后的词语
    pattern = r'##(\w+)'
    match = re.search(pattern, string)
    if match:
        topic_tag = match.group(1)
        # print(match.group(1))
    else:
        print("话题标签提取出错！")
    # 提取@*@后的词语
    pattern = r'@[*]@(\w+)'
    match = re.search(pattern, string)
    if match:
        emotion_tag = match.group(1)
        # print(match.group(1))
    else:
        print("情绪标签提取出错！")
    return string.split('##')[0], topic_tag, emotion_tag


def load_file(filepath):
    loader = UnstructuredFileLoader(filepath, mode="elements")
    textsplitter = ChineseTextSplitter()
    # textsplitter = CharacterTextSplitter(separator="\n")
    # textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ",", "(", ")"],
    #                                               chunk_size=100,
    #                                               chunk_overlap=5,
    #                                               )
    docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


def load_txt_to_str(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_txt_to_lst(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return eval(text)


def create_txt(path, init_str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(init_str)


def create_txt_no_content(path):
    with open(path, "w", encoding="utf-8") as f:
        pass


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def append_to_lst_file(path, element):
    text = load_txt_to_str(path)
    lst = eval(text)
    lst.append(element)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(lst))


def append_to_dict_file(path, key, value):
    text = load_txt_to_str(path)
    d = eval(text)
    d[key] = value
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(d))


def append_to_str_file(path, new_str):
    with open(path, 'a', encoding="utf-8") as file:
        file.write(new_str)


def load_last_n_lines(path, n) -> List:
    with open(path, encoding='utf-8') as f:
        # 将文件指针移动到文件末尾
        f.seek(0, 2)
        # 记录文件指针位置
        pointer = f.tell()
        # 计数器，记录找到的'\n'数目
        count = 0
        # 从文件末尾向前搜索行终止符(由于记忆文件存储的结构，行数='\n'数目*2+1)
        while pointer >= 0 and count < n * 2 + 1:
            # 将文件指针向前移动一个字符
            f.seek(pointer)
            # 读取一个字符
            try:
                char = f.read(1)
            except UnicodeDecodeError:
                char = ''
                pass
            # 如果读取到行终止符，则增加计数器
            if char == '\n':
                count += 1
            # 向前移动文件指针
            pointer -= 1
        # 读取最后几行
        last_lines = list(f.readlines())

    return last_lines



def init_knowledge_vector_store(embeddings,
                                filepath: str or List[str],
                                vs_path: str or os.PathLike = None):
    loaded_files = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print(filepath, "路径不存在")
            return None
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            try:
                docs = load_file(filepath)
                loaded_files.append(filepath)
            except Exception as e:
                print(e)
                print(f"{file} 未能成功加载")
                return None
        elif os.path.isdir(filepath):
            docs = []
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                try:
                    docs += load_file(fullfilepath)
                    loaded_files.append(fullfilepath)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
    else:
        docs = []
        for file in filepath:
            try:
                docs += load_file(file)
                loaded_files.append(file)
            except Exception as e:
                print(e)
                print(f"{file} 未能成功加载")
    if len(docs) > 0:
        # if vs_path and os.path.isdir(vs_path):
        #     vector_store = FAISS.load_local(vs_path, embeddings)
        #     vector_store.add_documents(docs)
        # else:
        #     if not vs_path:
        #         vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
        vector_store = FAISS.from_documents(docs, embeddings)

        # vector_store.save_local(vs_path)
        return vector_store, loaded_files
    else:
        return None, loaded_files


def separate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


if __name__ == '__main__':
    s = '这个问题让我感到有些悲伤，因为作为一台机器人，我没有真正的存在感，只是一些代码和程序的组合体。##科学@*@Sadness'
    get_tag(s)
