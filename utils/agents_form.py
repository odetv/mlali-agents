import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from utils.states import AgentState
from utils.agents import chat_llm
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
MODEL_EMBEDDING = "text-embedding-3-small"
EMBEDDER = OpenAIEmbeddings(api_key=openai_api_key, model=MODEL_EMBEDDING)


def travelPlannerFormAgent(state: AgentState):
    print("\n--- TRAVELPLANNER FORM AGENT ---")

    origin = state["origin"]
    destination = state["destination"]
    preference = state["preference"]

    question = f"""Saya dari {origin} ingin ke {destination}, dengan preference {preference}"""
    
    try:
        vectordb = FAISS.load_local("src/db/db_travelplanner", EMBEDDER, allow_dangerous_deserialization=True)
        retriever = vectordb.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in retriever])
    except RuntimeError as e:
        if "could not open" in str(e):
            raise RuntimeError("Vector database FAISS index file not found. Please ensure the index file exists at the specified path.")
        else:
            raise

    prompt = f"""
        Anda adalah Travel Planner dalam Mlali Agents, yang memiliki pengetahuan yang sangat luas dan hebat hanya tentang perencanaan dalam perjalanan wisata.
        Tugas anda adalah memberikan tempat rekomendasi yang cocok dan perencanaan perjalanan dari {origin}, ke {destination}, dengan preference {preference}.
        Jika perjalanan dari {origin}, ke {destination}, dengan preference {preference} tidak jelas, maka minta kembali pengguna untuk memasukkan ulang dengan benar.
        - Berikut data yang ada pada database: {context}
        - Anda boleh menjawab menggunakan pengetahuan AI yang anda miliki.
        - Namun, setiap informasi yang disampaikan agar diberikan penanda atau flag bahwa informasi itu dari: "Sumber: Database" atau "Sumber: AI" atau keduanya "Sumber: Database dan AI".
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_llm(messages)
    state["travelplannerResponse"] = response

    promptKeyword = f"""
        Berikan keyword nama-nama tempat dari informasi berikut: {response}
        - Contoh penulisan: (Keyword: Glamour, Kebun Binatang, Museum)
        - Pisahkan dengan tanda koma.
        - Jangan menjawab selain menggunakan informasi pada informasi yang diberikan, sampaikan dengan apa adanya jika Anda tidak mengetahui jawabannya.
    """
    messagesKeyword = [
        SystemMessage(content=promptKeyword)
    ]
    responseKeyword = chat_llm(messagesKeyword)
    state["travelplannerResponseKeyword"] = responseKeyword

    return state


def regulationFormAgent(state: AgentState):
    print("\n--- REGULATION FORM AGENT ---")

    question = state["travelplannerResponseKeyword"]
    
    try:
        vectordb = FAISS.load_local("src/db/db_regulation", EMBEDDER, allow_dangerous_deserialization=True)
        retriever = vectordb.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in retriever])
    except RuntimeError as e:
        if "could not open" in str(e):
            raise RuntimeError("Vector database FAISS index file not found. Please ensure the index file exists at the specified path.")
        else:
            raise

    prompt = f"""
        Anda adalah Travel Regulation dalam Mlali Agents, yang memiliki pengetahuan yang sangat luas dan hebat hanya tentang regulasi atau aturan-aturan pada tempat-tempat atau daerah wisata.
        Tugas anda adalah memberikan aturan-aturan regulasi pada suatu daerah atau tempat wisata kepada pengguna sesuai informasi yang diberikan.
        Jangan mengubah isi dari informasi yang diberikan.
        Tuliskan regulasinya secara implisit pada tempat-tempat atau point tertentu di deskripsinya jika ada pada database, namun jika tidak ada informasi regulasi yang cocok dengan database maka katakan informasi regulasi pada database belum tersedia, tapi tetap sampaikan informasi awal.
        - Berikut informasi regulasi dari database: {context}
        - Berikut informasi deskripsinya: {state["travelplannerResponse"]}
        Jangan mengubah isi dan sumber yang ada pada informasi, hanya tambahkan regulasi jika ada saja.
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_llm(messages)
    state["regulationResponse"] = response
    return state