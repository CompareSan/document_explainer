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
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
)

retaining_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

loader = PyPDFLoader("my_pdf.pdf")
pages = loader.load()

document_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)


docs = document_splitter.split_documents(pages)

db = Chroma.from_documents(pages, embeddings)

question_answering = ConversationalRetrievalChain.from_llm(
    llm, retriever=db.as_retriever(), memory=retaining_memory
)

tools = [
    Tool(
        name="Knowledge Base",
        func=question_answering.run,
        description=("use this tool when answering questions related to pdf file"),
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

while True:
    question = input("Enter your query: ")
    if question == "exit":
        break
    print(agent(question))
