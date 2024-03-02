import os

import streamlit as st
from dotenv import (
    find_dotenv,
    load_dotenv,
)
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    load_tools,
)
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

load_dotenv(find_dotenv())


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT")


@st.cache_resource
def build_agent(_docs: list[Document]) -> AgentExecutor:

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    retaining_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
    )
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(_docs, embeddings)
    retriever = db.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "pdf_search",
        "Search for information in the pdf document. \
          For any questions about the pdf document, you must use this tool!",
    )

    tools = load_tools(["serpapi"]) + [retriever_tool]

    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=retaining_memory,
        early_stopping_method="generate",
        max_iterations=3,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor
