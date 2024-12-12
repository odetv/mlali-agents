import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from utils.debug_time import time_check
from utils.states import AgentState
from utils.llm import LLM, EMBEDDER


def chat_llm(question: str):
    openai = LLM
    result = ""
    try:
        stream_response = openai.stream(question)
        for chunk in stream_response:
            token = chunk.content
            result += token
            print(token, end="", flush=True)
    except Exception as e:
        error = str(e)
        if "401" in error and "Incorrect API key" in error:
            raise ValueError("Incorrect API key provided. Please check your OpenAI API key.")
        else:
            raise e
    return result


@time_check
def assistantAgent(state: AgentState):
    print("\n--- ASSISTANT AGENT ---")

    promptTypeQuestion = """
        Anda adalah seoarang pemecah pertanyaan pengguna.
        Tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 3 konteks diajukan (Pilih hanya 1 konteks yang paling sesuai saja):
        - TRAVELGUIDE_AGENT - Jika pertanyaan mengacu pada tujuan wisata saja atau disebutkan juga dia darimana.
        - REGULATION_AGENT - Pertanyaan yang menyebutkan mengenai regulasi atau aturan-aturan yang perlu disiapkan di tempat wisata atau tempat-tempat tertentu yang vital.
        - GENERAL_AGENT - Ketika pertanyaan diluar nalar (berwisata ke luar angkasa, tempat yang tidak nyata dan lain-lain) tidak jelas dalam konteks mencari tempat wisata, dan tidak sesuai dengan konteks diatas. 
        Jawab pertanyaan dan sertakan pertanyaan pengguna dengan contoh seperti {"NAMA_AGENT": "pertanyaan pengguna"}.
        Buat dengan format data JSON tanpa membuat key baru.
    """
    messagesTypeQuestion = [
        SystemMessage(content=promptTypeQuestion),
        HumanMessage(content=state["question"]),
    ]
    responseTypeQuestion = chat_llm(messagesTypeQuestion).strip().lower()
    
    state["question_type"] = responseTypeQuestion
    print("\nPertanyaan:", state["question"])

    total_agents = 0
    if "general_agent" in state["question_type"]:
        total_agents += 1
    if "travelguide_agent" in state["question_type"]:
        total_agents += 2
    if "regulation_agent" in state["question_type"]:
        total_agents += 1

    state["totalAgents"] = total_agents
    print(f"DEBUG: Total agents bertugas: {state['totalAgents']}")

    pattern = r'"(.*?)":\s*"(.*?)"'
    matches = re.findall(pattern, responseTypeQuestion)
    result_dict = {key: value for key, value in matches}

    state["generalQuestion"] = result_dict.get("general_agent", None)
    state["travelguideQuestion"] = result_dict.get("travelguide_agent", None)
    state["regulationQuestion"] = result_dict.get("regulation_agent", None)
    
    print(f"DEBUG: generalQuestion: {state['generalQuestion']}")
    print(f"DEBUG: travelguideQuestion: {state['travelguideQuestion']}")
    print(f"DEBUG: regulationQuestion: {state['regulationQuestion']}")

    return state


@time_check
def generalAgent(state: AgentState):
    print("\n--- GENERAL AGENT ---")
    prompt = f"""
        Anda adalah Mlali Agents, yang memiliki pengetahuan yang sangat luas dan hebat hanya tentang wisata.
        Pertanyaan pengguna terkadang membingungkan, coba pahami dan berikan respons baik sehingga pengguna merasa nyaman.
        Jika membingungkan, rekomendasikan beberapa hal mengenai travel dalam berwisata.
        Jangan merespons hal-hal yang diluar dari konteks wisata.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["generalQuestion"])
    ]
    response = chat_llm(messages)
    state["responseFinal"] = response

    return state


@time_check
def travelGuideAgent(state: AgentState):
    print("\n--- TRAVELGUIDE AGENT ---")

    question = state["travelguideQuestion"]
    
    try:
        vectordb = FAISS.load_local("src/db/db_travelguide", EMBEDDER, allow_dangerous_deserialization=True)
        retriever = vectordb.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in retriever])
    except RuntimeError as e:
        if "could not open" in str(e):
            raise RuntimeError("Vector database FAISS index file not found. Please ensure the index file exists at the specified path.")
        else:
            raise

    prompt = f"""
        Anda adalah Travel Guide dalam Mlali Agents, yang memiliki pengetahuan yang sangat luas dan hebat hanya tentang pemandu perjalanan berwisata.
        Tugas anda adalah memberikan panduan perjalanan wisata kepada pengguna sesuai dengan konteks.
        - Berikut data yang ada pada database: {context}
        - Anda boleh menjawab menggunakan pengetahuan AI yang anda miliki.
        - Namun, setiap informasi yang disampaikan agar diberikan penanda atau flag bahwa informasi itu dari: "Sumber: Database" atau "Sumber: AI" atau keduanya "Sumber: Database dan AI".
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=question)
    ]
    response = chat_llm(messages)
    state["travelguideResponse"] = response

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
    state["travelGuideResponseKeyword"] = responseKeyword

    return state


@time_check
def regulationAgent(state: AgentState):
    print("\n--- REGULATION AGENT ---")

    travelguideResponse = state.get("travelguideResponse", "")
    if not travelguideResponse:
        travelguideResponse = ""

    question = f"""{state["question"]} {state["travelGuideResponseKeyword"]}"""
    
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
        - Berikut informasi deskripsinya: {travelguideResponse}
        Jangan mengubah isi dan sumber yang ada pada informasi, hanya tambahkan regulasi jika ada saja.
    """
    messages = [
        SystemMessage(content=prompt)
    ]

    response = chat_llm(messages)
    state["responseFinal"] = response
    
    return state



# @time_check
# def resultWriterAgent(state: AgentState):
#     if len(state["finishedAgents"]) < state["totalAgents"]:
#         print("\nMenunggu agent lain menyelesaikan tugas...")
#         return None
    
#     elif len(state["finishedAgents"]) == state["totalAgents"]:
#         info = "\n--- RESULT WRITER AGENT ---"
#         print(info)
#         prompt = f"""
#             Berikut pedoman yang harus diikuti untuk menulis ulang informasi:
#             - Berikan informasi secara lengkap dan jelas apa adanya sesuai informasi yang diberikan.
#             - Jangan menjawab selain menggunakan informasi pada informasi yang diberikan, sampaikan dengan apa adanya jika Anda tidak mengetahui jawabannya.
#             - Jangan tawarkan informasi lainnya selain informasi yang diberikan yang didapat saja.
#             - Jangan pernah mengubah seperti isi, sumber dan regulasi yang tercantum pada informasi, karena tugas anda adalah menulis ulang informasi.
#             - Hasilkan response dalam format Markdown.
#             Berikut adalah informasinya:
#             {state["answerAgents"]}
#         """
#         messages = [
#             SystemMessage(content=prompt),
#             HumanMessage(content=state["question"])
#         ]
#         response = chat_llm(messages)
#         print(response)
#         state["responseFinal"] = response

#         return {"responseFinal": state["responseFinal"]}