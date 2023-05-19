import re
import openai
import agent.useless as ul
from agent.utils import load_txt_to_str, append_to_str_file, CharacterInfo


class DialogEventGenerator:
    def __init__(self, model_name="gpt3_5", temperature=0.6):
        self.model_name = model_name
        if model_name == "gpt3_5":
            self.core_model = 'gpt-3.5-turbo-0301'
        elif model_name == "gpt3_5free":
            self.core_model = 'gpt-3.5-turbo'
        else:
            raise AttributeError("分类器模型选择参数出错！传入的参数为", model_name)

        self.temperature = temperature
        self.prompt = "用连贯的一句话描述以下对话存在的关键信息，要求包含时间、地点、人物（如果时间和地点不明确，则不需要提出来）：" \
                      "“{{{DIALOG}}}”"

    def send(self, massages):
        return openai.ChatCompletion.create(
            model=self.core_model,
            messages=massages,
            temperature=self.temperature
        )

    def send_free(self, prompt, sys_mes):
        return ul.Completion.create(prompt=prompt,
                                    systemMessage=sys_mes,
                                    parentMessageId="",
                                    temperature=self.temperature)

    def do(self, dialog_str):
        prompt = self.prompt.replace("{{{DIALOG}}}", dialog_str)
        if self.model_name == "gpt3_5":
            massages = [{"role": 'user', "content": prompt}]
            res = self.send(massages)
            ans = res.choices[0].message.content
        elif self.model_name == 'gpt3_5free':
            res = self.send_free(prompt, sys_mes='')
            ans = res['text']
        else:
            raise AttributeError("分类器模型选择参数出错！传入的参数为", self.model_name)
        return ans


# class EventGenerator:
#     def __init__(self, world_name, character_lst, model_name="gpt3_5", temperature=0.6):
#         super().__init__()
#         if model_name == "gpt3_5":
#             self.model_name = 'gpt-3.5-turbo'
#         elif model_name == "gpt3_5free":
#             self.model_name = model_name
#         else:
#             raise AttributeError("分类器模型选择参数出错！传入的参数为", model_name)
#
#         self.temperature = temperature
#         self.world_name = world_name
#         self.character_lst = character_lst
#         self.identity_file = 'agent/memory/' + self.world_name + '/global/all.txt'
#         self.event_file = 'agent/memory/' + self.world_name + '/local/{{{AI_NAME}}}/event.txt'
#         self.prompt = ''
#         self.init_prompt()
#
#     def init_prompt(self):
#         self.prompt = '"""'
#         self.prompt = load_txt_to_str(self.identity_file)
#         for c in self.character_lst:
#             self.prompt += ('\n' + c + '的事件：' + load_txt_to_str(self.event_file.replace("{{{AI_NAME}}}", c)))
#         self.prompt += '"""\n'
#         # print("len:", len(self.prompt))
#
#     def send(self, massages):
#         return openai.ChatCompletion.create(
#             model=self.model_name,
#             messages=massages,
#             temperature=self.temperature
#         )
#
#     def send_free(self, prompt, sys_mes):
#         return ul.Completion.create(prompt=prompt,
#                                     systemMessage=sys_mes,
#                                     parentMessageId="",
#                                     temperature=self.temperature)
#
#     def do(self, character_lst):
#         character_str = ",".join(character_lst)
#         example_str = "用以下方式描述出来。例如：\"\"\"小明事件：小明找小红去图书馆。" \
#                       "小红事件：小红去了博物馆。\"\"\"\n不要生成多余的句子或词语。每个人只生成一个事件，" \
#                       "每个事件可能只涉及一个人，也可能涉及多个人。"
#         massages = [{"role": 'user', "content": self.prompt}, {"role": 'assistant', "content": "我能帮你做些什么？"},
#                     {"role": 'user', "content": "猜想" + character_str + "接下来分别会做什么？表述成已经发生的事情。"
#                                                 + example_str}]
#         # print(massages)
#         if self.model_name == 'gpt-3.5-turbo':
#             res = self.send(massages)
#             ans = res.choices[0].message.content
#         elif self.model_name == 'gpt3_5free':
#             res = self.send_free(prompt=massages[2]['content'], sys_mes=massages[0]['content'])
#             ans = res['text']
#         events = self.event_to_dict(ans)
#         if events == {}:
#             print("事件未成功产生，可能是由于身份信息过少或不够详细。")
#             print("事件产生器的回复：", ans)
#             print("注：若事件产生器的回复已产生能理解的事件，但事件未录入，请重新产生一次。")
#             return
#         print("产生的事件：", events)
#         # 去重列表
#         tmp_cha_lst = []
#         for cha in events:
#             if cha not in tmp_cha_lst:
#                 tmp_cha_lst.append(cha)
#                 # 保存事件
#                 self.save_event(cha, events[cha])
#                 # 检查是否存在其他人物
#                 for c in character_lst:
#                     if c != cha and c in events[cha]:
#                         tmp_cha_lst.append(c)
#                         self.save_event(c, events[cha])
#
#     def save_event(self, character, event_str):
#         path = self.event_file.replace("{{{AI_NAME}}}", character)
#         append_to_str_file(path, event_str)
#
#     def event_to_dict(self, s):
#         # 使用正则表达式来匹配键值对
#         pattern = r"(\w+)事件：(.+?)(?=\n|$)"
#         # 使用正则表达式进行匹配
#         matches = re.findall(pattern, s)
#         # 将匹配结果转换成字典
#         d = dict(matches)
#         return d
