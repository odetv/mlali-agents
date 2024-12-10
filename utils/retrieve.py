from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def retrieve(query: str, db_path: str):
    db = FAISS.load_local("src/db/vector_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    query = "desa wisata di buleleng"
    relevant_response = db.similarity_search_with_relevance_scores(query, k=10)

    print(relevant_response)

    return relevant_response