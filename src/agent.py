import os

from dotenv import (
    find_dotenv,
    load_dotenv,
)
from langchain.agents import (
    Tool,
    initialize_agent,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv(find_dotenv())


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")


def build_agent(docs):

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(
        temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
    )

    retaining_memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True
    )

    db = Chroma.from_documents(docs, embeddings)

    question_answering = ConversationalRetrievalChain.from_llm(
        llm, retriever=db.as_retriever(), memory=retaining_memory
    )

    tools = [
        Tool(
            name="Knowledge Base",
            func=question_answering.run,
            description=(
                "use this tool when answering questions related to a document"
            ),
        )
    ]

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        memory=retaining_memory,
    )

    return agent
