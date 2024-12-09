
import os
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils.states import AgentState


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
MODEL_LLM = "gpt-4o-mini"
LLM = ChatOpenAI(api_key=openai_api_key, model=MODEL_LLM, temperature=0, streaming=True)
MODEL_EMBEDDING = "text-embedding-3-small"
EMBEDDER = OpenAIEmbeddings(api_key=openai_api_key, model=MODEL_EMBEDDING)


def chat_llm(question: str):
    result = LLM.invoke(question).content if hasattr(LLM.invoke(question), "content") else LLM.invoke(question)
    return result


def assistantAgent(state: AgentState):
    print("\n--- ASSISTANT AGENT ---")

    promptTypeQuestion = """
        Anda adalah seoarang pemecah pertanyaan pengguna.
        Tugas Anda sangat penting. Klasifikasikan atau parsing pertanyaan dari pengguna untuk dimasukkan ke variabel sesuai konteks.
        Tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 4 konteks diajukan (Pilih hanya 1 konteks yang paling sesuai saja):
        - GENERAL_AGENT - Pertanyaan yang berkaitan dengan hal umum atau tidak jelas.
        - TRAVELGUIDE_AGENT - Pertanyaan yang menyebutkan segala informasi mengenai perjalanan ingin kemana.
        - TRAVELPLANNER_AGENT - Pertanyaan yang menyebutkan mengenai darimana berasal, tujuan ingin kemana, dan preferensi perjalanan mengenai apa.
        - REGULATION_AGENT - Pertanyaan yang menyebutkan mengenai regulasi atau aturan-aturan yang diperlukan di tempat wisata.
        Jawab pertanyaan dan sertakan pertanyaan pengguna dengan contoh seperti ({"NAMAAGENT_AGENT": "Pertanyaannya"}).
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
        total_agents += 1
    if "travelplanner_agent" in state["question_type"]:
        total_agents += 1
    if "regulation_agent" in state["question_type"]:
        total_agents += 1

    state["totalAgents"] = total_agents
    print(f"DEBUG: Total agents bertugas: {state['totalAgents']}")

    pattern = r'"(.*?)":\s*"(.*?)"'
    matches = re.findall(pattern, responseTypeQuestion)
    result_dict = {key: value for key, value in matches}

    state["generalQuestion"] = result_dict.get("general_agent", None)
    state["travelguideQuestion"] = result_dict.get("travelguide_agent", None)
    state["travelplannerQuestion"] = result_dict.get("travelplanner_agent", None)
    state["regulationQuestion"] = result_dict.get("regulation_agent", None)
    
    print(f"DEBUG: generalQuestion: {state['generalQuestion']}")
    print(f"DEBUG: travelguideQuestion: {state['travelguideQuestion']}")
    print(f"DEBUG: travelplannerQuestion: {state['travelplannerQuestion']}")
    print(f"DEBUG: regulationQuestion: {state['regulationQuestion']}")

    return state



def generalAgent(state: AgentState):
    # pertanyaan umum
    return ""


def travelGuideAgent(state: AgentState):
    # travel guide
    return ""


def regulationAgent(state: AgentState):
    # agent yang mencari aturan2 di tempat wisata
    return ""


def travelPlannerAgent(state: AgentState):
    # input : origin, destination, preference
    return ""


def resultWriterAgent(state: AgentState):
    # travel guide
    return ""