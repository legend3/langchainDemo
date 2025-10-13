import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_models import ChatTongyi  # ✅ 改成通义千问
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig


"""
LangChain构建聊天机器人、流式输出的处理
总结: 
    1.一个用户可能打开多个「聊天」窗口。
    2.每一个窗口 = 一个 session_id（独立的会话上下文）。
    3.每个会话里，LangChain 都会用同一个MessagesPlaceholder(variable_name="xxx")来表示“用户这次说的话 + 历史上下文”。
        这样，模型就能在同一栏内记住上下文对话关系，而不同栏之间互不干扰。
"""


# ---------------- 创建模型 ----------------
# 你可以换成 qwen-turbo、qwen-plus、qwen-max 等
model = ChatTongyi(model="qwen-plus")

# ---------------- 定义提示模板 ----------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。请用{language}回答所有问题。"),
    MessagesPlaceholder(variable_name="my_msg")  # 2.提示模板的接口（告诉 LangChain：“我这里会放一段消息历史”）
])

# 得到链
chain = prompt_template | model

# ---------------- 聊天历史管理 ----------------
store = {}  # 用于存储所有会话的历史记录


def get_session_history(session_id: str):
    """
        一、session_id —— 用来区分「不同会话」
            🔹 作用：
            session_id 就像「聊天窗口的唯一编号」。
            LangChain 在 RunnableWithMessageHistory 里用它来识别「哪个用户的聊天记录」应该被加载或保存。
            比如：
                用户	            session_id	        聊天记录
                LaoXiao	        zs1234	        你好 → 我叫LaoXiao → 我的名字是什么？
                Lisa	        lis2323	        Hi → Tell me a joke
    """

    """根据 session_id 获取或新建聊天历史对象"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # 1.历史消息的存储（真正存储消息的地方（谁说了什么））
    return store[session_id]


# ---------------- 创建支持记忆的可运行对象 ----------------
"""
每次调用链时，（MessagesPlaceholder(variable_name="my_msg")）定义的这个字段(my_msg)标识里的消息会被插入到提示模板；
然后，RunnableWithMessageHistory 会把这些消息追加到当前 session_id 的历史里，从而形成多轮对话记忆。
"""
do_message = RunnableWithMessageHistory(    # 3.调度它们俩配合工作的胶水（每次调用时自动读取+更新历史，并把它插到模板里）
    chain,
    get_session_history,
    input_messages_key="my_msg"
)

# ---------------- 会话示例 ----------------
config = {"configurable": {"session_id": "zs1234"}}  # 当前会话ID（在实际项目中，通常不会固定写死，而是动态生成或获取 session_id）

# 第1轮
resp1 = do_message.invoke(
    {
        "my_msg": [HumanMessage(content="你好啊！ 我是LaoXiao")],
        "language": "中文"
    },
    config=config
)
print("第1轮：", resp1.content)

# 第2轮
resp2 = do_message.invoke(
    {
        "my_msg": [HumanMessage(content="请问：我的名字是什么？")],
        "language": "中文"
    },
    config=config
)
print("第2轮：", resp2.content)


# 流式输出
config = {"configurable": {"session_id": "lis2323"}}
print("\n流式输出：", end="")
for resp in do_message.stream(
    {
        "my_msg": [HumanMessage(content="请给我讲一个笑话？")],
        "language": "中文"
    },
    config=config
):
    print(resp.content, end="-")

print("\n✅ 流式输出结束")


# 准备三轮消息
config = {"configurable": {"session_id": "legend3"}}    #
messages = [
    "你好啊！我是LaoXiao",
    "请问你记得我吗？",
    "我的名字是什么？"
]

# 循环每轮对话
for i, msg in enumerate(messages, 1):
    print(f"\n第{i}轮：", end="")
    for resp in do_message.stream(
        {
                "my_msg": [HumanMessage(content=msg)],
                "language": "中文"
        },
        config=config
    ):
        print(resp.content, end="")
print("\n✅ 多轮流式输出结束")