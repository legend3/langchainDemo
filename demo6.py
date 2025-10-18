from operator import itemgetter
import pymysql
import bs4
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatTongyi  # ✅ 通义千问大模型


# ======================== PyMySQL 兼容 MySQLdb ========================
pymysql.install_as_MySQLdb()  # 🔑 关键

# ======================== 初始化模型 ========================
# 你可以换成 qwen-turbo、qwen-plus、qwen-max 等版本
model = ChatTongyi(model="qwen-plus")

# ======================== 数据库连接 ========================
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'test_db8'
USERNAME = 'root'
PASSWORD = 'root'

# 使用 PyMySQL 驱动，URI 保持 mysql+mysqldb://
MYSQL_URI = f"mysql+mysqldb://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
db = SQLDatabase.from_uri(MYSQL_URI)

# ======================== 构建链 ========================
# ① 使用通义模型自动生成 SQL
sql_gen_chain = create_sql_query_chain(model, db)

# ② 创建一个 SQL 执行工具
execute_sql_tool = QuerySQLDatabaseTool(db=db)

# ③ 定义回答模板（让模型根据 SQL 执行结果回答用户问题）
answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    请用简洁、准确的中文回答: """
)

# ④ 将多个组件组合成一个完整流程：
#     用户问题 → 生成SQL → 执行SQL → 模型生成自然语言答案
chain = (
    RunnablePassthrough.assign(query=sql_gen_chain)
    .assign(result=itemgetter('query') | execute_sql_tool)
    | answer_prompt
    | model
    | StrOutputParser()
)

# ======================== 测试 ========================
if __name__ == "__main__":
    question = "请问：员工表中有多少条数据？"
    response = chain.invoke({"question": question})
    print("✅ 答案：", response)
