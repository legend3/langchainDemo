import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import chat_agent_executor

# ======================== PyMySQL 兼容 MySQLdb ========================
import pymysql
pymysql.install_as_MySQLdb()  # 🔑 关键

# ======================== 环境配置 ========================
os.environ["DASHSCOPE_API_KEY"] = "你的通义千问API_KEY"  # ✅ 替换成你自己的通义API Key

# 如果之前开了代理，建议关闭
# os.environ.pop('http_proxy', None)
# os.environ.pop('https_proxy', None)

# ======================== 初始化模型 ========================
# 可选: qwen-turbo / qwen-plus / qwen-max
model = ChatTongyi(model="qwen-plus")

# ======================== 数据库连接 ========================
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'test_db8'
USERNAME = 'root'
PASSWORD = 'root'

# 使用 PyMySQL 驱动
MYSQL_URI = f"mysql+mysqldb://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
db = SQLDatabase.from_uri(MYSQL_URI)

# ======================== 创建工具和 Agent ========================
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

system_prompt = """
你是一个专门与 SQL 数据库交互的智能助手。
给定一个用户的问题，请生成正确的 SQL 查询并执行它，然后阅读查询结果并用中文简明地回答。

注意事项：
1. 如果用户没有指定数量，请限制最多 10 条结果。
2. 可以按相关列排序，以返回最匹配的记录。
3. 执行查询前请检查 SQL 的正确性。
4. 如果出错，请自动重写 SQL 并重试。
5. 严禁执行插入、更新、删除等写操作（仅限查询）。
6. 查询前请先查看数据库中可用的表结构。
"""

system_message = SystemMessage(content=system_prompt)

# ✅ 用通义千问模型创建 SQL Agent
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools, system_message)

# ======================== 测试对话 ========================
resp = agent_executor.invoke({'messages': [HumanMessage(content='哪个部门下面的员工人数最多？')]})

# ======================== 输出结果 ========================
result = resp["messages"]
print("\n================= 模型消息链 =================")
for i, msg in enumerate(result):
    print(f"[{i}] {msg}")

print("\n================= 最终答案 =================")
print(result[-1].content)
