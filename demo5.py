import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatTongyi         # 替换 ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings    # 替换 OpenAIEmbeddings


# ================= 加载网页内容 =================
loader = WebBaseLoader(
    web_paths=["https://www.cnblogs.com/superhin/p/17755515.html"],  # 确认正文在 HTML 中
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("forFlow"))
    ),
)


docs = loader.load()

# ================= 文本切分 =================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# ================= 向量化存储（Chroma） =================
# ✅ 使用 DashScopeEmbeddings 替换 OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=DashScopeEmbeddings(model="text-embedding-v2"))

# 检索器
retriever = vectorstore.as_retriever()

# ================= 主提示模板 =================
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ================= 模型创建 =================
# 可选模型：qwen-turbo（快）、qwen-plus（平衡）、qwen-max（最强）
model = ChatTongyi(model="qwen-plus")

# 创建文档问答链
chain1 = create_stuff_documents_chain(model, prompt)

# ================= 子链（历史感知检索器） =================
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ✅ 使用千问模型创建历史感知检索器
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# ================= 会话记忆（Memory） =================
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ================= 整合为完整链 =================
chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ================= 测试对话 =================
resp1 = result_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={"configurable": {"session_id": "zs123456"}},
)
print("🧠 第一次回答：", resp1["answer"])

resp2 = result_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "zs123456"}},  # ✅ 使用相同会话 ID 共享上下文
)
print("💬 第二次回答：", resp2["answer"])

