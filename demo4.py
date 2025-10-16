from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import chat_agent_executor

""" 搜索工具(DuckDuckGoSearchRun)、 """

# ---------------- 创建模型 ----------------
model = ChatTongyi(model="qwen-plus")

# ---------------- 创建搜索工具 ----------------
search = DuckDuckGoSearchRun()
tools = [search]

# ---------------- 创建 Agent代理：让大模型可以自动推理是否需要工具去完成用户的答案 ----------------
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

# ---------------- 测试 ----------------
print("🚀 测试开始...")

# 测试1：普通问题
resp = agent_executor.invoke({"messages": [HumanMessage(content="湖南的省会是哪个城市？")]})  # messages固定的！
print("\n🧠 普通问题测试：")
print("最终回答：", resp["messages"][-1].content)

# 测试2：需要搜索的问题
resp2 = agent_executor.invoke({"messages": [HumanMessage(content="今天长沙天气怎么样？")]})
print("\n🌦️ 搜索问题测试：")
print("\nmessages:", resp2["messages"])
print("最终回答：", resp2["messages"][-1].content)

print("\n✅ 测试完成！")