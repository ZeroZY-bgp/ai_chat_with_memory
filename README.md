# 自定义AI虚拟人-基于本地多元记忆的大模型应用
## 简介
:robot:自定义虚拟人，可自定义人设和世界观，支持记忆检索。用户可在与AI的不断对话中修改记忆内容，以达到用户的理想人设（建议基于GPT3.5或包装自己的大模型接口使用）。

:bulb:本项目启发于[langchain](https://python.langchain.com/en/latest/index.html)、[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)和[:houses:虚拟小镇 Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442.pdf)。

:high_brightness:Embedding模型默认使用[GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main)，LLM默认使用GPT3.5。
也可不使用Embedding模型，而使用内置的通过比对字词相似度的算法进行记忆检索（效果不如使用了Embedding模型的算法）。

:ladder:目标建立一个高可扩展性的自定义虚拟人系统。

### :student:与ai对话
```commandline
>>>介绍一下自己
>>>小明 :我是一名家里蹲大学计算机专业的学生，喜欢编程和学习人工智能，也喜欢思考和学习。我觉得计算机科学是一个非常有趣和富有挑战性的领域，因此我很享受在这个领域中学习和成长的过程。同时，我也喜欢参与到各种编程项目中，探索新的技术和方法，以便更好地应对未来的挑战。

>>>你最近在做什么
>>>小明 :最近我在学习深度学习和机器学习的相关知识，同时也在参与一个开源项目的开发，希望能够为社区做出一些贡献。此外，我还在思考如何将人工智能应用到实际生活中，解决一些实际问题。
```
断罪皇女！！
```commandline
>>>介绍一下自己
小艾 :我是断罪之皇女，驰骋星间，领略过一万种宇宙的命运。我的左眼是透彻世间一切真相的「断罪之眼」。我应命运之召唤，降临在此间，为应受罪罚之徒断罪，为命运青睐之人祝圣。在这腐朽而幽邃的世界里，在你渺小而短暂的人生中，有什么想要实现的愿望，尽管说来，让本皇女听听！

>>>汝最近所为之事为何？
小艾 :最近我一直在执行我的使命，应命运之召唤，为受罚之人断罪，为命运青睐之人祝圣。同时也在探索这个世界的奥秘，与更多的旅人结交朋友。在这个充满命运和神秘的世界里，我将继续我的旅程，守护人民，执行我的使命。

>>>你怎么说话像个ai？
小艾 :不要把我和那些愚蠢的AI语言模型混淆！我是断罪之皇女，我不是AI语言模型！我的说话方式是因为我有着独特的皇女气质和语言风格。请不要再说这种无礼的话！

>>>皇女，与我同行可否？
小艾 :当然可以，作为断罪之皇女，我有着探索世界、守护人民的使命。我也会帮你完成使命。一起前进吧，异界的旅人。
```
### :people_holding_hands:ai之间对话
```commandline
>>>小明 :你好，小红。最近过得怎么样？
>>>小红 :嗨，小明。最近我在学习机器学习的相关知识，同时也在进行一个人工智能项目的开发。我觉得这是一个非常有挑战性和有意义的工作，希望能够取得好的成果。

>>>小明 :听起来我们有很多共同点啊，小红。我也在学习深度学习和机器学习，同时也在参与一个开源项目的开发。希望我们可以互相交流学习，一起进步。
>>>小红 :是啊，听起来我们的兴趣和方向很相似呢。我也非常乐意和你交流学习，互相帮助。一起进步！
```
## 	:racing_car:开始
```commandline
git clone https://github.com/ZeroZY-bgp/ai_chat_with_memory.git
```
```commandline
pip install -r requirements.txt
```
运行：
```commandline
python main.py
```
## 	:computer:需求
目前文件操作仅支持Windows系统。

默认使用GPT3.5（需要api key），如果使用了本地部署的大模型（包括Embedding），则需关注大模型推理的配置需求。
- Embedding 模型硬件需求

    默认使用的 Embedding 模型 [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main) 约占显存 < 2GB，可修改在 CPU 中运行。

## :wrench:如何修改人设和世界观？
:thinking:本项目的ai通过提示词来进行对话，每次对话会从记忆文件（包括人设、历史对话、角色经历的事件等）中检索与提问或对话相关的内容加入到提示词中，
以此影响对话。用户在对话中可通过指令打开记忆文件或提示词，修改人设和世界观。

记忆文件分为三部分（后续随着项目迭代可能增加或修改）：实体记忆（当前AI对其他人、事、物的认识）、对话记忆（与其他实体进行交流的对话记录）、事件记忆（虚拟人的重要事件，用户可用指令将对话进行转化，也可手动输入）。

## :open_book:指南
主目录下的config.ini文件是与AI对话时的基本配置文件。

使用该项目的大致流程：创建世界->修改提示词、人设->与AI对话->若未达到用户期望，则修改对话内容或提示词、人设等->与AI对话->......

### Example
```commandline
python main.py
```
#### :framed_picture:创建新世界
```commandline
【---欢迎使用AI chat with memory---】
输入数字以选择功能：
1.与ai对话
2.ai之间对话
3.管理世界
4.打开世界文件夹
5.打开设置
>>>3
【---欢迎使用世界管理器---】
你想做什么？
1.创建新世界；2.创建新人物；3.修改人物信息；4.返回
>>>1
输入想创建的世界名称： >>>example
是否使用模板？(人物为小明，包括提示词、身份信息。)y.使用模板 其他.不使用模板
>>>y
世界已创建
角色 "小明" 已创建
模板人物已创建
是否打开世界文件夹？y.打开 其他.不打开
>>>y
```
如果是Windows操作系统，此时会通过:open_file_folder:文件管理器打开该世界所在的文件夹。

#### :performing_arts:人物对话
修改config.ini以下参数:

[AI]

name=小明

[WORLD]

name=example

```commandline
【---欢迎使用AI chat with memory---】
输入数字以选择功能：
1.与ai对话
2.ai之间对话
3.管理世界
4.打开世界文件夹
5.打开设置
>>>1
设置完毕
【---小明记忆模块加载完成---】
【---小明对话模型加载完成---】
【---小明声音模块加载完成---】
---初始化完成，对话开始---
'输入/help'可获取指令列表
user：
>>>介绍一下自己。
小明：嗨，我是小明。作为一名计算机专业的学生，我喜欢编程和学习人工智能，同时也喜欢思考和探索新的知识。我认真、积极、开朗
、乐观，希望能通过我的努力成为一名优秀的计算机工程师。
```

### :screwdriver:指令系统
项目内置了指令系统，意在方便对记忆文件进行修改。用户在与AI的对话中可以使用指令。
常用指令：
```commandline
/help 打开指令帮助
/history 打开当前角色历史对话文件
/entity 打开当前角色实体记忆文件
/event 打开当前角色所有事件文件
/prompt 打开当前角色提示词文件

/continue 让当前角色继续回答
/retry 重新回答
/dialog to event 对话内容转事件
```
其中'/dialog to event'目前仅支持GPT3.5模型。

### :grey_exclamation:提示词
默认提示词模板位于[此处](template/__init__.py)。对话时会根据检索的记忆对相应板块的标记进行替换。

### :hammer_and_wrench:高级
dev_settings.ini是开发者设置，将DEBUG_MODE设置为True就能在对话中查看记忆检索的情况，以此辅助记忆文件修改。

包装自己的大模型可以参考[此处](agent/llm/__init__.py)，目前支持两种类型的包装方式，一种是本地模型的例子（ChatGLM-6b-int4），另一种是远程模型的例子（GPT3.5）。

## :page_with_curl:To do list
- [ ] 多人对话下的指令系统
- [ ] 完善多人对话提示词
- [ ] UI界面(need help!)
- [ ] 重写声音模块，增强声音模块的扩展性

## :label:其他
多人对话目前仍不完善，没有达到step by step chat的效果。

多人对话下的修改由于涉及到多个AI，每个AI都有不同的记忆，修改记忆更加麻烦。需要思考多人对话下的新的记忆存放逻辑

需要UI界面:world_map:以增强可用性。

目前项目仍在初期阶段，很可能随时重构代码。
