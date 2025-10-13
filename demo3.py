from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatTongyi  # ✅ 替换 ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings  # ✅ 替换 OpenAIEmbeddings

""" 构建文档和向量空间、检索器和模型结合 """

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

# ---------------- 1.向量存储（Chroma） ----------------
# ✅ 替换成通义千问的 Embedding 模型
vector_store = Chroma.from_documents(   # “把文字变成数学坐标，让计算机能够"理解"文字之间的相似性。”（向量坐标是嵌入模型根据其训练学到的语义关系来确定的，确保语义相似的文本在向量空间中位置接近！）
    documents,
    embedding=DashScopeEmbeddings(model="text-embedding-v2")  # 通义 embedding 模型
)

# ---------------- 检索器 ----------------
""" 相似度的查询：返回相似的分数，分数越低相似度越高 """
# print(vector_store.similarity_search_with_score("咖啡猫"))

# 检索器：bind(k=1)返回相似度最高的第一个
# RunnableLambda将retriever转成Runable对象，才能在chain链中使用
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)
# 批量检索多个向量
# print(retriever.batch(["请介绍一下猫？"]))


# ---------------- 2.Prompt 模板：（将向量空间与大语言模型结合起来） ----------------
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
chain = {
    "question": RunnablePassthrough(),  # RunnablePassthrough：让用户输入继续传递下去（允许我们将用户的问题之后再传递给prompt和model）# ← 接收整个输入
    "context": retriever    # ← 接收整个输入
} | prompt_temp | model    # Chain 结构：单一输入流

# ---------------- 执行测试 ----------------
resp = chain.invoke("请介绍一下猫？")  # RunnablePassthrough 和 retriever 都接收 chain.invoke() 的整个输入，因此 chain.invoke() 只能传入一个值（两个组件都接收相同的完整输入）

print("\n🤖 通义千问回答：\n", resp.content)
