# 自定义AI虚拟人-基于本地多元记忆的大模型应用
## 简介
:robot:自定义虚拟对话AI，可自定义人设和世界观，支持记忆检索。用户可在与AI的不断对话中修改记忆内容，以达到用户的理想人设（建议基于GPT3.5或包装自己的大模型接口使用）。

:bulb:本项目启发于[langchain](https://python.langchain.com/en/latest/index.html)、[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)和[虚拟小镇:houses: Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442.pdf)。
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
## 开始
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

默认使用GPT3.5（需要api key），如果使用了本地部署的大模型，则需关注大模型推理的配置需求。

## :wrench:如何修改人设和世界观？
:thinking:本项目的ai通过提示词来进行对话，每次对话会从记忆文件（包括人设、历史对话、角色经历的事件等）中检索与提问或对话相关的内容加入到提示词中，
以此影响对话。用户在对话中可通过指令打开记忆文件或提示词，修改人设和世界观。

## :open_book:指南
主目录下的config.ini文件是与AI对话时的基本配置文件。

该项目使用的大致流程：创建世界->修改提示词、人设->与AI对话->若未达到用户期望，则修改对话内容或提示词、人设等->与AI对话->...

### example
>>>python main.py
> 