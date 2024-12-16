import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
MODEL_LLM = "gpt-4o-mini"
LLM = ChatOpenAI(api_key=openai_api_key, model=MODEL_LLM, temperature=0, streaming=True)
MODEL_EMBEDDING = "text-embedding-3-small"
EMBEDDER = OpenAIEmbeddings(api_key=openai_api_key, model=MODEL_EMBEDDING)