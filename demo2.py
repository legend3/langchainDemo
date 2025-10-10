import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_models import ChatTongyi  # ✅ 改成通义千问
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig


# ---------------- 创建模型 ----------------
# 你可以换成 qwen-turbo、qwen-plus、qwen-max 等
model = ChatTongyi(model="qwen-plus")

# ---------------- 定义提示模板 ----------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。请用{language}回答所有问题。"),
    MessagesPlaceholder(variable_name="my_msg")
])

# 得到链
chain = prompt_template | model

# ---------------- 聊天历史管理 ----------------
store = {}  # 用于存储所有会话的历史记录


def get_session_history(session_id: str):
    """根据 session_id 获取或新建聊天历史对象"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ---------------- 创建支持记忆的可运行对象 ----------------
do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="my_msg"
)

# ---------------- 会话示例 ----------------
config = {"configurable": {"session_id": "zs1234"}}  # 当前会话ID

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

# 第3轮（流式输出）
config2 = {"configurable": {"session_id": "lis2323"}}
print("第3轮（流式输出）：", end="")
for resp in do_message.stream(
    {
        "my_msg": [HumanMessage(content="请给我讲一个笑话？")],
        "language": "English"
    },
    config=config2
):
    print(resp.content, end="-")

print("\n✅ 流式输出结束")
