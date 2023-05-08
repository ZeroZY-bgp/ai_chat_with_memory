import re

import openai
import copy
from agent.utils import get_tag, read_txt_to_str, append_to_lst_file

tag_list = ['闲聊', '翻译', '专业知识']
emotion_list = ['Joy', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Neutral']


class Classifier:

    def __init__(self, temperature=0.6):
        super().__init__()
        self.model_name = "gpt-3.5-turbo-0301"
        # self.max_tokens = max_tokens
        self.temperature = temperature
        content_s = "你现在是一个句子话题分类器，为每一句话提取话题标签“##标签”和情绪标签“@*@emotion”，" \
                    "话题标签仅有##闲聊##翻译##专业知识，情绪标签仅有以下几种：" \
                    "Joy、Sadness、Anger、Fear、Disgust、Neutral。" \
                    "你每次只回答“##标签@*@emotion”。" \
                    "这是例子：句子：好啊，谢谢你！我很期待。你回复：##日常对话@*@Joy" \
                    "若理解了请回复“abc”。"
        self.default_massage = {"role": 'system', "content": content_s}
        ans = ''
        # while ans != 'abc':
        #     self.massages = [self.default_massage]
        #     res = self.send(self.massages)
        #     ans = res.choices[0].message.content
        #     print('分类器握手回答：', ans)
        self.massages = [self.default_massage]
        res = self.send(self.massages)
        ans = res.choices[0].message.content
        print('分类器握手回答：', ans)
        self.massages.append({"role": 'assistant', "content": ans})

    def send(self, massages):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=massages,
            temperature=self.temperature,
        )

    def do(self, string):
        cur_massages = copy.deepcopy(self.massages)
        # ask
        new_ask = {"role": "user", "content": string}
        cur_massages.append(new_ask)
        res = self.send(cur_massages)
        ans = res.choices[0].message.content
        ans_, topic_tag, emotion_tag = get_tag(ans)
        while topic_tag not in tag_list:
            print("话题标签分类错误")
            print("标签为:", topic_tag)
            tip_str = "话题标签只有3种，没有" + topic_tag
            topic_tag, emotion_tag = self.correct_tag(tip_str, cur_massages)

        while emotion_tag not in emotion_list:
            print("情绪标签分类错误")
            print("标签为:", emotion_tag)
            tip_str = "情绪标签只有6种，没有" + emotion_tag
            topic_tag, emotion_tag = self.correct_tag(tip_str, cur_massages)

        return topic_tag, emotion_tag

    def correct_tag(self, tip_str, cur_massages):
        new_ask = {"role": "user", "content": tip_str}
        cur_massages.append(new_ask)
        res = self.send(cur_massages)
        ans = res.choices[0].message.content
        ans_, topic_tag, emotion_tag = get_tag(ans)
        return topic_tag, emotion_tag


class EventDetector:
    def __init__(self, temperature=0.001):
        super().__init__()
        self.model_name = "gpt-3.5-turbo"
        self.temperature = temperature
        self.prompt = "人物列表：[Alice、Lisa、Bob]你现在是一个人物交互判别器，" \
                      "只有在人物列表中的人物与其他同在列表中的人物有过互动才会被视为有互动，否则为无互动。" \
                      "一个人物对其他人物的评价视为无互动。例子如下：" \
                      "“Alice说：我给了Lisa一块蛋糕。”回答：Y。理由：Alice和Lisa都存在于人物列表中。" \
                      "“Lisa说：我去找Lisa玩。”回答：N。理由：只出现Lisa一个人物。" \
                      "“Bob说：我去找Jake爬山。”你回答：N。理由：Jake不存在于人物列表中。" \
                      "“Lisa说：我每天都和Alice一起读书，我觉得她很好学。”回答：Y。理由：Lisa每天都和Alice一起读书。" \
                      "“Lisa说：我觉得Alice很好学。”回答：N。理由：这是Lisa对Alice的评价，视为无互动。" \
                      "你仅回复“Y”（表示有）或“N”（表示无），不用给出理由。" \
                      "询问：“{{{context}}}”"

    def send(self, massages):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=massages,
            temperature=self.temperature,
        )

    def do(self, string):
        prompt = self.prompt.replace("{{{context}}}", string)
        massages = [{"role": 'user', "content": prompt}]
        res = self.send(massages)
        ans = res.choices[0].message.content
        if 'N' in ans:
            return False
        elif 'Y' in ans:
            return True
        else:
            print('分类器未分类')
            return False


class EventGenerator:
    def __init__(self, world_name, character_lst, multi_event=False, temperature=0.6):
        super().__init__()
        self.model_name = "gpt-3.5-turbo"
        self.temperature = temperature
        self.world_name = world_name
        self.character_lst = character_lst
        self.multi_event = multi_event
        self.identity_file = 'agent/memory/' + self.world_name + '/global/all.txt'
        self.event_file = 'agent/memory/' + self.world_name + '/local/{{{AI_NAME}}}/event.txt'
        self.prompt = ''
        self.init_prompt()

    def init_prompt(self):
        self.prompt = '"""'
        self.prompt = read_txt_to_str(self.identity_file)
        for c in self.character_lst:
            self.prompt += ('\n' + c + '的事件：' + read_txt_to_str(self.event_file.replace("{{{AI_NAME}}}", c)))
        self.prompt += '"""\n'
        # print(self.prompt)
        print("len:", len(self.prompt))

    def send(self, massages):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=massages,
            temperature=self.temperature
        )

    def do(self, character_lst):
        character_str = ",".join(character_lst)
        if self.multi_event:
            example_str = "用1、2...N的方式罗列出来。例如：Alice事件：1.Alice找Lisa去图书馆。2.Alice读书。" \
                          "Lisa事件：1.Lisa去了博物馆。2.Lisa和Alice去了自然公园。\n每个事件可能只有一个人，也可能有多个人。" \
                          "每个人的事件数目不多于3个。"
        else:
            example_str = "用以下方式描述出来。例如：Alice事件: Alice找Lisa去图书馆。" \
                          "Lisa:Lisa去了博物馆。\n每个事件可能只有一个人，也可能有多个人。"
        massages = [{"role": 'user', "content": self.prompt}, {"role": 'assistant', "content": "我能帮你做些什么？"},
                    {"role": 'user', "content": "猜想" + character_str + "接下来分别会做什么？表述成已经发生的事情。"
                     + example_str}]
        print(massages)
        res = self.send(massages)
        ans = res.choices[0].message.content
        print(ans)
        events = self.event_to_dict(ans)
        print(events)
        # 去重列表
        tmp_cha_lst = []
        if not self.multi_event:
            for cha in events:
                if cha not in tmp_cha_lst:
                    tmp_cha_lst.append(cha)
                    # 保存事件
                    self.save_event(cha, events[cha])
                    # 检查是否存在其他人物
                    for c in character_lst:
                        if c != cha and c in events[cha]:
                            tmp_cha_lst.append(c)
                            self.save_event(c, events[cha])

    def save_event(self, character, event_str):
        path = self.event_file.replace("{{{AI_NAME}}}", character)
        append_to_lst_file(path, event_str)

    def event_to_dict(self, ans):
        if self.multi_event:
            # 定义正则表达式模式
            pattern = r'(\w+)事件：\n([\d\D]*?)(?=\n\w+事件：|\Z)'

            # 使用正则表达式进行匹配
            matches = re.findall(pattern, ans)

            # 将匹配结果转换为字典
            events = {match[0]: re.findall(r'\d+\. (.+)', match[1]) for match in matches}
        else:
            # 定义正则表达式模式
            pattern = r'(\w+)事件：([\w\W]+?)(?=\n\w+事件：|\Z)'

            # 使用正则表达式进行匹配
            matches = re.findall(pattern, ans)

            # 将匹配结果转换为字典
            events = {match[0]: match[1].strip() for match in matches}
        return events
