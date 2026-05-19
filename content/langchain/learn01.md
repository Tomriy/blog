+++
date = '2026-05-19T11:20:20+08:00'
draft = false
title = 'LangChain 学习之旅：从核心概念到实战应用'
language = "cn-zh"
author = 'Torimy'
tags= ["LangChain", "LLM", "Python", "AI", "教程"]
+++


## 引言

大语言模型（LLM）的能力令人惊叹，但直接调用 API 往往难以构建复杂的、具备上下文感知和推理能力的应用。**LangChain** 应运而生，它是一套专门为简化 LLM 应用开发而设计的框架，让你能像搭积木一样组合出强大的智能体。

本文将以**新手友好**的视角，带你一步步理解 LangChain 的核心模块，并通过一个完整的问答机器人示例，完成从理论到实践的跨越。

---

## 1. 为什么是 LangChain？

- **标准化接口**：统一了不同模型（OpenAI、Anthropic、本地模型）和向量数据库的调用方式。
- **组件化设计**：将应用拆分为提示、链、记忆、检索等模块，职责清晰，易于复用。
- **强大的编排能力**：通过 `Chain` 和 `Agent` 实现多步推理、工具调用，而不仅仅是单轮对话。
- **生态丰富**：官方和社区提供了大量 `Tool`、`Loader`、`Retriever` 集成，开箱即用。

> 注意：2026 年 LangChain 已进入成熟期，推荐使用 `langchain-core` 配合 `langchain-openai` 等包，体验最新的 LCEL（LangChain 表达式语言）特性。

---

## 2. 核心组件速览

我们先用一张图理解 LangChain 的运转逻辑，然后逐一拆解。

```
用户输入 → Prompt模板 → 语言模型 → 输出解析器 → 结果
↑ ↓
Memory(记忆) Retriever(检索)
↑ ↓
Agent(代理) ←→ Tools(工具)
```


### 2.1 模型 I/O：与 LLM 对话的“翻译官”

这是最基础的模块，包含三个要素：

1. **提示模板（Prompt Template）**：将动态变量插入到预设的提示中。
2. **语言模型（Language Model）**：统一调用各类 LLM。
3. **输出解析器（Output Parser）**：将模型返回的非结构化文本变成程序可读的格式（如 JSON）。

**示例：**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个擅长讲冷笑话的助手"),
    ("user", "讲一个关于{topic}的笑话")
])

# 2. 模型
model = ChatOpenAI(model="gpt-4o", temperature=0.8)

# 3. 输出解析器
parser = StrOutputParser()

# 将它们串联
chain = prompt | model | parser
print(chain.invoke({"topic": "程序员"}))
```

### 2.2 检索增强生成（RAG）：让 LLM 拥有“外脑”
LLM 的训练数据是静态的，想让模型基于最新文档或私有知识回答，就需要 RAG 流水线。

典型流程：加载文档 → 文本切割 → 向量化 → 存入向量数据库 → 检索相关内容 → 注入提示。

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 加载与切割
loader = TextLoader("./knowledge.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 向量化并存入 Chroma
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# 构建检索器
retriever = vectorstore.as_retriever()
```

将检索器接入链：

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

answer = rag_chain.invoke("What did the author say about...")
```

### 2.3 链与记忆：不止于单次对话
**Chain** 将多个步骤串联成工作流。早期有 `LLMChain`，现在更推荐使用 **LCEL** (| 管道) 构建，它天然支持流式、异步和并行。

**Memory** 负责在多次交互中保持状态。现代 LangChain 应用中，我们通常使用 `RunnableWithMessageHistory` 包装链，把历史消息存入外部存储
```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}  # 实际项目中可换为 Redis

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_history_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 第一次对话
response1 = with_history_chain.invoke(
    {"question": "我叫小明"},
    config={"configurable": {"session_id": "user123"}}
)
# 第二次对话，模型能记住名字
response2 = with_history_chain.invoke(
    {"question": "我叫什么名字？"},
    config={"configurable": {"session_id": "user123"}}
)
```

### 2.4 代理（Agent）与工具：让模型学会使用“手脚”
当你需要模型**自主决策调用何种工具**（搜索、计算器、查询数据库）时，Agent 就登场了。

**Tools**：封装了具体功能的接口（如 `tool` 装饰器）。

**Agent**：根据 LLM 的推理，选择工具并观察结果，直到得出最终答案。

现代写法推荐使用 `create_react_agent`：

```python
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
agent = create_react_agent(model, tools)

result = agent.invoke({"messages": [("user", "2026年温网男单冠军是谁？")]})
```

## 3. 实战：构建一个可对话的知识库助手
结合 RAG、记忆和 Streaming，打造一个完整的问答应用。
```python
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ---- 初始化向量存储（仅首次） ----
if "vectorstore" not in st.session_state:
    # 假设已有文档加载并切割
    st.session_state.vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )

# ---- 构建链 ----
system_prompt = """你是智能助手，基于以下上下文回答问题。如果不知道，就说不知道，不要编造。
上下文：{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

retriever = st.session_state.vectorstore.as_retriever()
model = ChatOpenAI(model="gpt-4o", streaming=True)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# ---- 添加对话记忆 ----
msgs = StreamlitChatMessageHistory(key="langchain_messages")
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# ---- Streamlit 界面 ----
st.title("📚 知识库问答助手")
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("请输入你的问题"):
    st.chat_message("human").write(question)
    config = {"configurable": {"session_id": "any"}}
    with st.chat_message("ai"):
        response = st.write_stream(
            chain_with_history.stream({"question": question}, config)
        )
```
这短短几十行代码就实现了一个拥有长记忆、基于私有知识、且流式输出的对话界面。


## 4. 学习建议与最佳实践

1、**从 LCEL 入手**：抛弃旧的 LLMChain，| 管道语法可读性强，且便于调试和组合。

2、**善用 LangSmith**：官方提供的调试平台，可以可视化每一步的输入输出，迅速定位问题。

3、**理解 Runnable 协议**：所有组件都是 Runnable，支持 invoke、stream、batch，掌握它就能轻松切换同步/异步。

4、**拆分独立组件**：不要写一个巨大的链，而是将检索、格式化、回答拆成独立 Runnable，便于测试和复用。

5、**控制 Token 消耗**：合理设置 chunk_size，使用 trim_messages 辅助函数管理历史长度。

6、**关注官方 Cookbook**：https://python.langchain.com/ 上有大量最新示例。


## 5. 结语
LangChain 为 LLM 应用开发提供了一套强大的抽象，它的真正价值在于让你**专注于业务逻辑而非底层胶水代码**。记住：框架只是工具，关键在于你对问题域的拆解和提示词的设计。

建议你今天就从一个小项目开始——也许是自己的个人知识库、也许是自动化邮件助手——亲自动手感受 LangChain 的威力。