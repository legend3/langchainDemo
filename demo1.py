from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi   # ✅ 用通义千问替代 ChatOpenAI
from langserve import add_routes


""" 采用LangChain调用LLM、提示模版、部署你的langchain程序 """

# ----------------------------
# 创建模型
# ----------------------------
# 可选模型名称："qwen-turbo"、"qwen-plus"、"qwen-max"
model = ChatTongyi(model="qwen-plus")

# ----------------------------
# 测试一：直接消息调用
# ----------------------------
msg = [
    SystemMessage(content="请将以下的内容翻译成意大利语"),
    HumanMessage(content="你好，请问你要去哪里？")
]

# result = model.invoke(msg)
# print(result.content)

# ----------------------------
# 测试二：带输出解析的 chain
# ----------------------------
parser = StrOutputParser()

""" system是起个头规范大模型后续用户的输入内容该怎么去回答，user是用户具体输入的内容问题，assistant是辅助大模型让大模型具有上下文记忆性 """
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "请将下面的内容翻译成{language}"),
    ("user", "{text}")
])

chain = prompt_template | model | parser

# 直接运行测试
print(chain.invoke({    # Chain 结构：函数式参数映射
                    "language": "English",  # 从输入字典提取特定字段
                    "text": "我下午还有一节课，不能去打球了。"}))  # 从输入字典提取特定字段

# ----------------------------
# 部署成 FastAPI 服务
# ----------------------------
app = FastAPI(
    title="我的Langchain服务",
    version="V1.0",
    description="使用通义千问翻译任何语句的服务器"
)

add_routes(
    app,
    chain,
    path="/chainDemo",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)   # http://localhost:8000/chainDemo/playground/   http://localhost:8000/docs  POST http://localhost:8000/chainDemo/invoke


