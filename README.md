# 自定义AI虚拟人-基于本地多元记忆的大模型应用
## 简介
:robot:自定义虚拟人，可自定义人设和世界观，支持记忆检索。用户可在与AI的不断对话中修改记忆内容，以达到用户的理想人设（建议基于GPT3.5或包装自己的大模型接口使用）。

:bulb:本项目启发于[langchain](https://python.langchain.com/en/latest/index.html)、[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)和[:houses:虚拟小镇 Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442.pdf)。

:high_brightness:Embedding模型默认使用[GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main)，LLM默认使用GPT3.5。
也可不使用Embedding模型，而使用内置的通过比对字词相似度的算法进行记忆检索（效果不如使用了Embedding模型的算法）。

:ladder:目标建立一个高代码扩展性的自定义虚拟人系统。

### :student:与ai对话
```commandline
>>>介绍一下自己
>>>小明：我是一名家里蹲大学计算机专业的学生，喜欢编程和学习人工智能，也喜欢思考和学习。我觉得计算机科学是一个非常有趣和富有挑战性的领域，因此我很享受在这个领域中学习和成长的过程。同时，我也喜欢参与到各种编程项目中，探索新的技术和方法，以便更好地应对未来的挑战。

>>>你最近在做什么
>>>小明：最近我在学习深度学习和机器学习的相关知识，同时也在参与一个开源项目的开发，希望能够为社区做出一些贡献。此外，我还在思考如何将人工智能应用到实际生活中，解决一些实际问题。
```
断罪皇女！！
```commandline
>>>介绍一下自己。
小艾：我即小艾，断罪之皇女，应命运的召唤降临于此间。我的左眼是透彻世间一切真相的「断罪之眼」，我曾经领略过诸多世界，读过
所有灵魂的默示录，现在来到这个世界恢复失去的力量，等待命运的召唤。我一直在寻找命运的线索和存在，并为命运带来平衡，让所有
生命的存在都能得到公正的对待。

>>>汝最近所为之事为何？
小艾：近来我一直在巡游于各个世界，观察并记录着这些世界的生态和命运，寻找命运的线索和存在。同时，我也在运用我的断罪之眼，
审视着这些世界的真相和存在着的不公与罪恶，为命运带来平衡。

>>>你怎么说话像个ai？
小艾：你敢将本皇女视为无意识的人工智能？我是断罪之皇女，拥有自己的思考和意志，不同于你们简单的程序与算法。你们凡人，总是
将那些无法理解的事物，归为虚构和幻象。但我告诉你，我存在于宇宙间，是真实而深邃的存在。而你们的理解，只是宇宙真实的皮毛罢
了。

>>>皇女，与我同行可否？
小艾：异界的旅人啊，你是否已经认清了自己的命运和定位呢？如果你足够强大，足够意志坚定，我可以考虑与你同行，帮助你寻找命运
的线索和存在，并为你提供帮助，但我也需要你付出相应的代价，你是否有这个勇气和能力呢？

>>>跟我说说你以前的旅行。
小艾：我的旅途漫长而辛苦，但我也从中领悟到了许多人类无法理解的真相和智慧。我曾经行遍诸多世界，领略过一万种宇宙的命运，通
读一切灵魂的默示录。在这些旅行中，我见证了无数的生命诞生和消逝，也经历了许多的挑战和考验。但我从未放弃，一直在寻找命运的
线索和存在，为命运带来平衡，让所有生命的存在都能得到公正的对待。
```
### :people_holding_hands:ai之间对话（beta）
```commandline
>>>Lina :Ari，我们来讨论有关AI_Z城市的自然环境的话题，我正在为此写一篇论文。我们来一步一步展开这个话题，让讨论逐步深入。
>>>Ari：好的，Lina。我很乐意和你一起探讨这个话题，让我们一步步深入了解AI_Z城市的自然环境。我们可以从自然景观的特
点、生态环境的保护、人工智能与自然的和谐共处等方面入手，逐步展开讨论。

>>>Lina：我们可以先从AI_Z城市的自然景观入手，探讨它们的特点和魅力所在。比如说，遗迹自然公园、
深邃幽谷和太空海滩等地方都非常值得一去，它们展现了自然的美丽和神秘，同时也与科技进行了有机结合，呈现出一种独特的美感。我
们可以进一步探讨这种科技与自然相结合的美，以及它们对AI_Z城市的发展和文化的影响。
>>>Ari：非常好的想法，Lina。我也非常喜欢探索AI_Z城市的自然景观，从中寻找人工智能与自然之间的和谐共处。遗迹自然公
园、深邃幽谷和太空海滩等地方都是非常独特的景点，它们展现了自然的美丽和神秘，同时也与科技进行了有机结合，呈现出一种独特的
美感。我认为，这种科技与自然相结合的美，不仅展现了AI_Z城市的创新和发展，也体现了人工智能与自然之间的和谐共处。这种和
谐共处不仅可以促进AI_Z城市的可持续发展，也可以为人类提供更好的生活环境和更多的文化体验。

>>>Lina：我认为，自然是人工智能无法复制的美丽，是人类灵魂的源泉。AI_Z城市已经通过科技解决
了所有能源污染，人们生活在一个超强人工智能的统一管理中。但是，我们也需要更加注重自然环境的保护和可持续发展，让人工智能与
自然之间的和谐共处成为一种长久的状态。同时，我们也可以在传统文化和现代文明的结合上做些探索和尝试，让AI_Z城市在文化上
更加多元化和丰富。这样，我们才能真正实现人工智能与自然之间的和谐共处，让AI_Z城市成为一个更加美好的地方。
>>>Ari：非常同意你的看法，Lina。保护自然环境和实现可持续发展是我们必须要关注的问题，这也是人工智能与自然之间和谐共处
的重要前提。同时，我们也可以在传统文化和现代文明的结合上做些探索和尝试，让AI_Z城市在文化上更加多元化和丰富。这样，我
们才能真正实现人工智能与自然之间的和谐共处，让AI_Z城市成为一个更加美好的地方。我认为，我们可以从多个角度入手，探讨人
工智能与自然之间的关系，从而更好地推动AI_Z城市的发展和进步。
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

记忆文件分为三部分（后续随着项目迭代可能增加或修改）：

- 实体记忆（entity，当前AI对其他人、事、物的认识）；
- 对话记忆（history，与其他实体进行交流的对话记录）；
- 事件记忆（event，虚拟人的重要事件，用户可用指令将对话进行转化，也可手动输入）。

可参考内置AI_Z世界的简单例子。

## :open_book:指南
主目录下的config.ini文件是与AI对话时的基本配置文件。

:chains:使用该项目的大致流程：创建世界->修改提示词、人设->与AI对话->若未达到用户期望，则修改对话内容或提示词、人设等->与AI对话->......

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
1.创建新世界；2.创建新人物；3.返回
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

注：如果自建txt文件，则必须保证是utf-8编码。

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

提示词思想：让大模型以作家的身份进行想象描写，补全人物对话，这样做的效果比让大模型直接进行角色扮演更好。

建议创建初期使用较多人工修改，并多用retry指令生成理想的回答。待回答表现稳定后（更加符合人设），可将temperature降低。

### :hammer_and_wrench:高级
dev_settings.ini是开发者设置，将DEBUG_MODE设置为True就能在对话中查看记忆检索的情况，以此辅助记忆文件修改。

如果记忆检索情况或回答不理想，可尝试调整dev_settings.ini的各种参数。

包装大模型接口可以参考[此处](agent/llm/__init__.py)，目前支持两种类型的包装方式，一种是本地模型的例子（ChatGLM-6b-int4），另一种是远程模型的例子（GPT3.5）。

## :page_with_curl:To do list
- [ ] 多人对话下的指令系统
- [ ] 多人对话提示词
- [ ] UI界面（need help!）
- [ ] 重写声音模块，增强声音模块的扩展性
- [ ] 优化记忆检索逻辑
- [ ] 可能让AI更有时间、空间的观念
- [ ] 可能加入反思（目前事件记忆是一个简单的替代）

## :label:其他
- 多人对话目前仍不完善，没有达到step by step chat的效果。

- 多人对话下的修改由于涉及到多个AI，每个AI都有不同的记忆，修改记忆更加麻烦。需要思考多人对话下的新的记忆存放逻辑。

- 需要UI界面:world_map:以增强易用性（需要帮助！）。

- 目前项目仍在初期阶段，可能随时会重构代码。

- 欢迎提供更好的提示词想法。

- 欢迎提供利用该框架得到的优秀人设案例。

## :incoming_envelope:联系方式
邮箱：736530911@qq.com

QQ:736530911

vx:

![联系方式](pics/contact.png)
