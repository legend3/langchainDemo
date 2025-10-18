import pymysql
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

# ======================== PyMySQL 兼容 MySQLdb ========================
pymysql.install_as_MySQLdb()

# ======================== 初始化模型 ========================
model = ChatTongyi(model="qwen-plus")

# ======================== 数据库连接 ========================
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'test_db8'
USERNAME = 'root'
PASSWORD = 'root'

MYSQL_URI = f"mysql+mysqldb://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
db = SQLDatabase.from_uri(MYSQL_URI)

# ======================== 系统提示 ========================
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""

# ======================== 用户问题 ========================
user_question = "哪个部门下面的员工人数最多？"

# ======================== 生成 SQL 查询 ========================
prompt_messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_question)
]

# 让 LLM 生成 SQL
llm_response = model.invoke(prompt_messages)
sql_query = llm_response.content

# ======================== 执行 SQL 查询 ========================
# 简单提取 SQL 语句（假设 LLM 返回的就是 SQL，如果有多余文字可自己处理）
sql_query = sql_query.strip()
print(sql_query)
if not sql_query.lower().startswith("select"):
    raise ValueError("生成的 SQL 语句不安全，不允许执行。")

# 加 LIMIT 10
if "limit" not in sql_query.lower():
    sql_query += " LIMIT 10"

print("执行 SQL:", sql_query)
results = db.run(sql_query)

print("\n查询结果:")
for row in results:
    print(row)
