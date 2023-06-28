import ast
import copy
import os

import textdistance
from langchain import FAISS
from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List

SENTENCE_SIZE = 100


# 代码来源:https://github.com/imClumsyPanda/langchain-ChatGLM/blame/master/textsplitter/chinese_text_splitter.py
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


class AnswerFragmentTextSplitter(CharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text_lst: List[str], text_splitter) -> List[str]:
        all_text = ''
        for text in text_lst:
            all_text += text
        new_lst = text_splitter.split_text(all_text)
        # 移除空字符串
        lst = [e for e in new_lst if e]
        return lst


class LineSplitter:

    @staticmethod
    def split_text(text) -> List[str]:
        words = text.split('\n')
        # 去掉空白字符串
        words = [word for word in words if word]
        return words


class SentenceSplitter(CharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text) -> List[str]:
        # 中英文标点符号
        punctuations = r'''！？。；!?.;'''
        # words = re.split('[' + punctuations + ']', text)
        words = re.findall(f'[^{punctuations}]*[{punctuations}]?', text)

        # 去掉空白字符串
        words = [word for word in words if word]
        return words


def fragment_text(text_lst, text_splitter):
    return AnswerFragmentTextSplitter().split_text(text_lst, text_splitter)


def high_word_similarity_text_filter(agent, mem_lst):
    # 字词相似度比对，算法复杂度o(n^2)
    remaining_memory = copy.deepcopy(mem_lst)  # 创建一个副本以避免在迭代时修改原始列表
    for i in range(len(mem_lst)):
        for j in range(len(mem_lst)):
            if i != j:
                str_i = mem_lst[i]
                str_j = mem_lst[j]
                sim_score = textdistance.jaccard(str_i, str_j)
                if sim_score > agent.dev_config.word_similarity_threshold:
                    # 如果两个字符串的字词相似度超过阈值，则删除较短的字符串（较短的字符串信息含量大概率较少）
                    del_e = mem_lst[i] if len(str_i) < len(str_j) else mem_lst[j]
                    if del_e in remaining_memory:
                        remaining_memory.remove(del_e)
    return remaining_memory


class BoundTextFilter:

    def __init__(self, interval_str):
        self.lower_bound_type = 'closed' if interval_str[0] == '[' else 'open'
        self.upper_bound_type = 'closed' if interval_str[-1] == ']' else 'open'
        interval = ast.literal_eval(interval_str.strip('[]()'))
        self.lower_bound = interval[0]
        self.upper_bound = interval[1]

    def _compare(self, upper_bound: bool, open_bound: bool, num: float):
        if upper_bound:
            return num < self.upper_bound if open_bound else num <= self.upper_bound
        else:
            return num > self.lower_bound if open_bound else num >= self.lower_bound

    def compare(self, num):
        # 判断是否在区间内
        lob = self.lower_bound_type == 'open'
        uob = self.upper_bound_type == 'open'
        return self._compare(upper_bound=False, open_bound=lob, num=num) \
            and self._compare(upper_bound=True, open_bound=uob, num=num)

    def filter(self, query, docs):
        mem_lst = []
        for t in docs:
            sim_score = textdistance.jaccard(query, t.page_content)
            if self.compare(sim_score):
                mem_lst.append(t)
        return mem_lst


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return None

    def top(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            return None

    def clear(self):
        self.stack.clear()

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

    def get_lst(self):
        return self.stack


class SimpleTextFilter:

    def __init__(self, top_k):
        self.top_k = top_k

    def filter(self, query, docs):
        min_num_stack = Stack()
        tmp_stack = Stack()
        min_num_stack.push({'text': '', 'score': 1.0})
        for t in docs:
            sim_score = textdistance.ratcliff_obershelp(query, t)
            if sim_score > min_num_stack.top()['score']:
                while sim_score > min_num_stack.top()['score']:
                    tmp_stack.push(min_num_stack.pop())
                min_num_stack.push({'text': t, 'score': sim_score})
                while not tmp_stack.is_empty() and min_num_stack.size() - 1 < self.top_k:
                    min_num_stack.push(tmp_stack.pop())
                tmp_stack.clear()
            else:
                if min_num_stack.size() - 1 < self.top_k:
                    min_num_stack.push({'text': t, 'score': sim_score})
        if min_num_stack.size() > 1:
            final_lst = []
            stack_lst = min_num_stack.get_lst()[1:]
            for e in stack_lst:
                final_lst.append(e['text'])
            return final_lst
        else:
            return []


class EntityTextFragmentFilter:

    def __init__(self, top_k, tsp, entity_weight=0.1):
        self.tsp = tsp
        self.top_k = top_k
        self.entity_weight = entity_weight

    @staticmethod
    def get_dict_with_scores(query, entity_dict):
        new_dict = {}
        # 对实体名相似程度进行打分
        for name, describe in entity_dict.items():
            new_dict[name] = {}
            new_dict[name]['score'] = textdistance.ratcliff_obershelp(query, name)
            new_dict[name]['text'] = describe

        return new_dict

    def filter(self, query, entity_dict):
        # ---给实体相似度打分
        # entity_dict_with_score = self.get_dict_with_scores(query, entity_dict)
        # ---
        # ---打碎实体描述文本
        # entity_mem = []
        # for name, tur in entity_dict_with_score.items():
        #     # 实体打碎策略：先切分，后罗列
        #     describe_lst = self.tsp.split_text(tur['text'])  # 切分实体描述
        #     for d in describe_lst:  # 罗列
        #         entity_mem.append({'text': name + ':' + d, 'entity_score': tur['score']})
        # # ---

        # ---打碎实体描述文本
        entity_mem = []
        for name, describe in entity_dict.items():
            # 实体打碎策略：先切分，后罗列
            describe_lst = self.tsp.split_text(describe)  # 切分实体描述
            for d in describe_lst:  # 罗列
                entity_mem.append(name + ':' + d)
        # ---

        min_num_stack = Stack()
        tmp_stack = Stack()
        min_num_stack.push({'text': '', 'score': 1.0})
        for e in entity_mem:

            # # 计算方式：描述相似度 * 字符串权重 + 实体相似度 * 实体权重
            # lcs_sim_score = textdistance.ratcliff_obershelp(query, e['text'])
            # sim_score = lcs_sim_score * (1 - self.entity_weight) + e['entity_score'] * self.entity_weight

            sim_score = textdistance.ratcliff_obershelp(query, e)

            if sim_score > min_num_stack.top()['score']:
                while sim_score > min_num_stack.top()['score']:
                    tmp_stack.push(min_num_stack.pop())
                min_num_stack.push({'text': e, 'score': sim_score})
                while not tmp_stack.is_empty() and min_num_stack.size() - 1 < self.top_k:
                    min_num_stack.push(tmp_stack.pop())
                tmp_stack.clear()
            else:
                if min_num_stack.size() - 1 < self.top_k:
                    min_num_stack.push({'text': e, 'score': sim_score})
        if min_num_stack.size() > 1:
            final_lst = []
            stack_lst = min_num_stack.get_lst()[1:]
            for e in stack_lst:
                final_lst.append(e['text'])
            return final_lst
        else:
            return []


class EntityVectorStoreFragmentFilter:

    def __init__(self, top_k, tsp, entity_weight=0.8):
        self.tsp = tsp
        self.top_k = top_k
        self.entity_weight = entity_weight

    @staticmethod
    def get_entity_names(entity_mem):
        entity_names = []
        for entity in entity_mem:
            # 可能有中文冒号，统一替换为英文冒号
            entity_name = entity.replace('：', ':')
            k, v = entity_name.split(":", 1)
            entity_names.append(k)
        return entity_names

    def filter(self, query, entity_dict, embeddings):
        # ---打碎实体描述文本
        entity_mem = []
        for name, describe in entity_dict.items():
            # 实体打碎策略：先切分，后罗列
            describe_lst = self.tsp.split_text(describe)  # 切分实体描述
            for d in describe_lst:  # 罗列
                entity_mem.append(name + ':' + d)
        # ---
        vs = FAISS.from_texts(entity_mem, embeddings)
        entity_with_score = vs.similarity_search_with_score(query, self.top_k)
        res_lst = []
        for doc in entity_with_score:
            res_lst.append(doc[0].page_content)

        return res_lst
