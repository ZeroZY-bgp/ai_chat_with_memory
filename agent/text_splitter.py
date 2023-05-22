import copy

import textdistance
from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List

SENTENCE_SIZE = 100


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > SENTENCE_SIZE:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > SENTENCE_SIZE:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > SENTENCE_SIZE:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls


class AnswerTextSplitter(CharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        # 包含中文和英文标点符号的正则表达式
        # punctuation = r'[。！？；，、‘’“”（）\[\]【】《》：!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
        punctuation = r'[。！？；，!?;,.]'
        # 使用正则表达式分割字符串
        segments = re.split(punctuation, text)
        # 移除空字符串
        segments = [segment for segment in segments if segment]
        return segments


def low_semantic_similarity_text_filter(agent, mem_lst):
    # 去掉低相似度搜索结果
    if len(mem_lst) == 0:
        return mem_lst
    max_score = mem_lst[-1].metadata["score"]
    truncate_i = 0
    for i, mem in enumerate(mem_lst):
        if mem.metadata["score"] > max_score - agent.semantic_similarity_threshold:
            truncate_i = i
            break
    return mem_lst[truncate_i:]


def high_word_similarity_text_filter(agent, mem_lst):
    # 字词相似度比对，算法复杂度o(n^2)
    remaining_memory = copy.deepcopy(mem_lst)  # 创建一个副本以避免在迭代时修改原始列表
    for i in range(len(mem_lst)):
        for j in range(len(mem_lst)):
            if i != j:
                str_i = mem_lst[i].page_content
                str_j = mem_lst[j].page_content
                sim_score = textdistance.jaccard(str_i, str_j)
                if sim_score > agent.word_similarity_threshold:
                    # 如果两个字符串的字词相似度超过阈值，则删除较短的字符串（较短的字符串信息含量大概率较少）
                    del_e = mem_lst[i] if len(str_i) < len(str_j) else mem_lst[j]
                    if del_e in remaining_memory:
                        remaining_memory.remove(del_e)
    return remaining_memory
