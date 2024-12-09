from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo")

response =  llm.invoke("Saya mau liburan ke bali, berikan rekomendasi")

print(response)