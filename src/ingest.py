from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(file_path: str):

    loader = PDFPlumberLoader(file_path)
    pages = loader.load()

    document_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

    document = document_splitter.split_documents(pages)

    return document
