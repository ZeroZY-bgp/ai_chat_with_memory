import datetime
import os
import re
from typing import List, Tuple

import numpy as np
from langchain import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema import Document

from agent.chinese_text_splitter import ChineseTextSplitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")


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
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        # textsplitter = CharacterTextSplitter(separator="\n")
        # textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ",", "(", ")"],
        #                                               chunk_size=1000,
        #                                               chunk_overlap=2,
        #                                               )
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


def read_txt_to_str(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def append_to_lst_file(path, element):
    text = read_txt_to_str(path)
    lst = eval(text)
    lst.append(element)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(lst))


def init_knowledge_vector_store(embeddings,
                                filepath: str or List[str],
                                vs_path: str or os.PathLike = None):
    loaded_files = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
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
        if vs_path and os.path.isdir(vs_path):
            vector_store = FAISS.load_local(vs_path, embeddings)
            vector_store.add_documents(docs)
        else:
            if not vs_path:
                vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
            vector_store = FAISS.from_documents(docs, embeddings)

        vector_store.save_local(vs_path)
        return vs_path, loaded_files
    else:
        print("文件均未成功加载")
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


def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4,
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)
    for j, i in enumerate(indices[0]):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i]
        doc = self.docstore.search(_id)
        id_set.add(i)
        docs_len = len(doc.page_content)
        for k in range(1, max(i, store_len - i)):
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        break_flag = True
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
            if break_flag:
                break
    id_list = sorted(list(id_set))
    id_lists = separate_list(id_list)
    for id_seq in id_lists:
        for id in id_seq:
            if id == id_seq[0]:
                _id = self.index_to_docstore_id[id]
                doc = self.docstore.search(_id)
            else:
                _id0 = self.index_to_docstore_id[id]
                doc0 = self.docstore.search(_id0)
                doc.page_content += doc0.page_content
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        docs.append((doc, doc_score))
    return docs


if __name__ == '__main__':
    s = '这个问题让我感到有些悲伤，因为作为一台机器人，我没有真正的存在感，只是一些代码和程序的组合体。##科学@*@Sadness'
    get_tag(s)
