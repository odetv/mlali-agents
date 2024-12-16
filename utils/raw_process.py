import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm import EMBEDDER


def processRegulation():
    print("Processing regulation...\n")
    DATASETS_DIR="src/datasets/data_regulation"
    VECTORDB_DIR="src/db/db_regulation"
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
    if not os.path.exists(VECTORDB_DIR):
        os.makedirs(VECTORDB_DIR)

    loader = PyPDFDirectoryLoader(DATASETS_DIR)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[" ", ",", ".", "\n", "\n\n", "\n\n\n", "\f"]
    )
    chunks = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(chunks, EMBEDDER)
    vectordb.save_local(VECTORDB_DIR)


def processTravelGuide():
    print("Processing travel guide...\n")
    DATASETS_DIR="src/datasets/data_travelguide"
    VECTORDB_DIR="src/db/db_travelguide"
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
    if not os.path.exists(VECTORDB_DIR):
        os.makedirs(VECTORDB_DIR)

    loader = PyPDFDirectoryLoader(DATASETS_DIR)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[" ", ",", ".", "\n", "\n\n", "\n\n\n", "\f"]
    )
    chunks = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(chunks, EMBEDDER)
    vectordb.save_local(VECTORDB_DIR)


def processTravelPlanner():
    print("Processing travel planner...\n")
    DATASETS_DIR="src/datasets/data_travelplanner"
    VECTORDB_DIR="src/db/db_travelplanner"
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
    if not os.path.exists(VECTORDB_DIR):
        os.makedirs(VECTORDB_DIR)

    loader = PyPDFDirectoryLoader(DATASETS_DIR)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[" ", ",", ".", "\n", "\n\n", "\n\n\n", "\f"]
    )
    chunks = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(chunks, EMBEDDER)
    vectordb.save_local(VECTORDB_DIR)


if __name__ == "__main__":
    processRegulation()
    processTravelGuide()
    processTravelPlanner()