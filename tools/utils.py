import os
import re
from typing import List

import openai
from langchain.document_loaders import UnstructuredFileLoader

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")


class CharacterInfo:
    def __init__(self, world_name, ai_name):
        self.world_name = world_name
        self.ai_name = ai_name
        self.folder_path = 'memory/' + self.world_name + '/' + self.ai_name
        self.prompt_path = self.folder_path + '/prompt' + self.ai_name + '.txt'
        self.history_path = self.folder_path + '/history' + self.ai_name + '.txt'
        self.entity_path = self.folder_path + '/entity' + self.ai_name + '.txt'
        self.event_path = self.folder_path + '/event' + self.ai_name + '.txt'
        self.traits_path = self.folder_path + '/traits' + self.ai_name + '.txt'


def openai_moderation(history, query):
    history_lst = []
    for dialog in history:
        history_lst.append(dialog[0] + dialog[1])
    history_lst.append(query)
    res = openai.Moderation.create(
        input=history_lst
    )
    for i, r in enumerate(res["results"]):
        if r["flagged"]:
            print(r)
            print(history_lst[i])
            return True
    return False


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


def load_docs(filepath, textsplitter):
    if not os.path.exists(filepath):
        print(filepath, "路径不存在")
        return None
    elif os.path.isfile(filepath):
        file = os.path.split(filepath)[-1]
        try:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load_and_split(text_splitter=textsplitter)
        except Exception as e:
            print(e)
            print(f"{file} 未能成功加载")
            return None
    else:
        print(filepath, "未能成功加载")
        return None
    return docs


def load_txt(filepath, textsplitter):
    if not os.path.exists(filepath):
        print(filepath, "路径不存在")
        return None
    elif os.path.isfile(filepath):
        try:
            txt = load_txt_to_str(filepath)
            txt = textsplitter.split_text(txt)
        except Exception as e:
            print(e)
            print(f"{filepath} 未能成功加载")
            return None
    else:
        print(filepath, "未能成功加载")
        return None
    return txt


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


def delete_last_line(file_path):
    with open(file_path, 'r+', encoding='utf-8') as file:
        file.seek(0, os.SEEK_END)
        position = file.tell() - 2
        count = 0
        while position > 0:
            file.seek(position)
            try:
                char = file.read(1)
            except UnicodeDecodeError:
                char = ''
            if char == '\n':
                count += 1
                if count >= 2:
                    file.seek(position + 1)
                    file.truncate()
                    break
            position -= 1


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
