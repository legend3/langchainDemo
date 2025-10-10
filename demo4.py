import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi  # ✅ 替换 ChatOpenAI
from langchain_community.embeddings import TongyiEmbeddings  # ✅ 替换 OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor
from langserve import add_routes



# ---------------- 创建模型（通义千问） ----------------
# 可用模型："qwen-turbo"（快速） / "qwen-plus"（平衡） / "qwen-max"（最强）
model = ChatTongyi(model="qwen-plus")

# ---------------- 创建 Tavily 搜索工具 ----------------
search = TavilySearchResults(max_results=2)  # 只返回两个搜索结果
tools = [search]

# ---------------- 创建 Agent（具有自动工具调用能力） ----------------
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

# ---------------- 测试 1：不需要搜索的普通问题 ----------------
resp = agent_executor.invoke({
    "messages": [HumanMessage(content="中国的首都是哪个城市？")]
})
print("\n🧠 问题：中国的首都是哪个城市？")
for m in resp["messages"]:
    print(f"角色：{m.type} ｜ 内容：{m.content}")

# ---------------- 测试 2：需要调用搜索工具的问题 ----------------
resp2 = agent_executor.invoke({
    "messages": [HumanMessage(content="北京天气怎么样？")]
})
print("\n🌦️ 问题：北京天气怎么样？")
for m in resp2["messages"]:
    print(f"角色：{m.type} ｜ 内容：{m.content}")

# 输出最后的回答内容
print("\n✅ 最终回答：", resp2["messages"][-1].content)
