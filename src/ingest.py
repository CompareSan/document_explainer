import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document


@st.cache_data
def load_document(file_path: str) -> list[Document]:
    loader = PDFPlumberLoader(file_path)
    pages = loader.load()

    document_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

    document = document_splitter.split_documents(pages)

    return document
