import os

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatTongyi  # ✅ 替换 ChatOpenAI
from langchain_community.embeddings import TongyiEmbeddings  # ✅ 替换 OpenAIEmbeddings
from langserve import add_routes


# ---------------- 通义千问配置 ----------------
# ⚠️ 替换为你自己的 DashScope API key
os.environ["DASHSCOPE_API_KEY"] = "你的通义千问API Key"

# ---------------- 创建模型 ----------------
# 可选模型: qwen-turbo, qwen-plus, qwen-max
model = ChatTongyi(model="qwen-plus")

# ---------------- 准备测试文档数据 ----------------
documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

# ---------------- 向量存储（Chroma） ----------------
# ✅ 替换成通义千问的 Embedding 模型
vector_store = Chroma.from_documents(
    documents,
    embedding=TongyiEmbeddings(model="text-embedding-v2")  # 通义 embedding 模型
)

# ---------------- 检索器 ----------------
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# ---------------- Prompt 模板 ----------------
message = """
你是一个知识丰富的宠物专家。
请使用以下上下文信息来回答问题：
上下文：
{context}

问题：
{question}

请用中文回答。
"""

prompt_temp = ChatPromptTemplate.from_messages([("human", message)])

# ---------------- 构建链（RAG） ----------------
# RunnablePassthrough：让用户输入继续传递下去
chain = {
    "question": RunnablePassthrough(),
    "context": retriever
} | prompt_temp | model

# ---------------- 执行测试 ----------------
resp = chain.invoke("请介绍一下猫？")

print("\n🤖 通义千问回答：\n", resp.content)
