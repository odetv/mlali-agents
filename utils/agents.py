
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
        Tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 4 konteks diajukan (Pilih hanya 1 konteks yang paling sesuai saja):
        - GENERAL_AGENT - Pertanyaan mengenai sapaan. 
        - TRAVELGUIDE_AGENT - Pertanyaan mengenai pengguna ingin melakukan perjalanan atau liburan.
        - TRAVELPLANNER_AGENT - Pertanyaan mengenai rencana perjalanan dengan menyebutkan darimana berasal (origin), tujuan ingin kemana (destination), dan preferensi perjalanan mengenai apa (preference).
        - REGULATION_AGENT - Pertanyaan yang menyebutkan mengenai regulasi atau aturan-aturan yang diperlukan di tempat wisata.
        Jawab pertanyaan dan sertakan pertanyaan pengguna dengan contoh seperti ({"GENERAL_AGENT": "Pertanyaannya"} atau {"TRAVELGUIDE_AGENT": "Pertanyaannya"} atau {"TRAVELPLANNER_AGENT": "Pertanyaannya"} atau {"REGULATION_AGENT": "Pertanyaannya"}).
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
    print("\n--- GENERAL AGENT ---")
    prompt = f"""
        Anda seorang yang memiliki pengetahuan yang sangat luas dan hebat tentang destinasi wisata.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["generalQuestion"])
    ]
    response = chat_llm(messages)
    agentOpinion = {
        "answer": response
    }
    print("\n\nGENERAL ANSWER:::", response)
    state["finishedAgents"].add("general_agent")
    return {"answerAgents": [agentOpinion]}


def travelGuideAgent(state: AgentState):
    print("\n--- TRAVELGUIDE AGENT ---")
    prompt = f"""
        Anda seorang yang memiliki pengetahuan yang sangat luas dan hebat tentang destinasi wisata.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["travelguideQuestion"])
    ]
    response = chat_llm(messages)
    print("\n\nTRAVELGUIDE ANSWER:::", response)
    state["finishedAgents"].add("tarvelguide_agent")
    state["travelguideResponse"] = response
    return state


def regulationAgent(state: AgentState):
    print("\n--- REGULATION AGENT ---")

    travelguideResponse = state.get("travelguideResponse", "")
    travelplannerResponse = state.get("travelplannerResponse", "")

    # If either one is None or empty, set them to ""
    if not travelguideResponse:
        travelguideResponse = ""
    if not travelplannerResponse:
        travelplannerResponse = ""

    prompt = f"""
        Anda seorang yang memiliki pengetahuan yang sangat luas dan hebat tentang destinasi wisata.
        Ini konteksnya: {travelguideResponse} {travelplannerResponse}
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_llm(messages)
    agentOpinion = {
        "answer": response
    }
    print("\n\nREGULATION ANSWER:::", response)
    state["finishedAgents"].add("regulation_agent")
    return {"answerAgents": [agentOpinion]}


def travelPlannerAgent(state: AgentState):
    # input : origin, destination, preference
    print("\n--- TRAVELPLANNER AGENT ---")
    prompt = f"""
        Anda seorang yang memiliki pengetahuan yang sangat luas dan hebat tentang destinasi wisata.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["travelplannerQuestion"])
    ]
    response = chat_llm(messages)
    print("\n\nTRAVELPLANNER ANSWER:::", response)
    state["finishedAgents"].add("travelplanner_agent")
    state["travelplannerResponse"] = response
    return state


def resultWriterAgent(state: AgentState):
    if len(state["finishedAgents"]) < state["totalAgents"]:
        print("\nMenunggu agent lain menyelesaikan tugas...")
        return None
    
    elif len(state["finishedAgents"]) == state["totalAgents"]:
        info = "\n--- RESULT WRITER AGENT ---"
        print(info)
        prompt = f"""
            Berikut pedoman yang harus diikuti untuk menulis ulang informasi:
            - Berikan informasi secara lengkap dan jelas apa adanya sesuai informasi yang diberikan.
            - Urutan informasi sesuai dengan urutan pertanyaan.
            - Jangan menyebut ulang pertanyaan secara eksplisit.
            - Jangan menjawab selain menggunakan informasi pada informasi yang diberikan, sampaikan dengan apa adanya jika Anda tidak mengetahui jawabannya.
            - Jangan tawarkan informasi lainnya selain informasi yang diberikan yang didapat saja.
            - Hasilkan response dalam format Markdown.
            Berikut adalah informasinya:
            {state["answerAgents"]}
        """
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["question"])
        ]
        response = chat_llm(messages)
        print(response)
        state["responseFinal"] = response

        return {"responseFinal": state["responseFinal"]}