from langgraph.graph import END, START, StateGraph
from utils.agents import assistantAgent, chat_llm, generalAgent, regulationAgent, resultWriterAgent, travelGuideAgent, travelPlannerAgent
from utils.create_graph_image import get_graph_image
from utils.debug_time import time_check
from utils.states import AgentState
from langchain_core.messages import HumanMessage, SystemMessage


@time_check
def runModel(question):
    workflow = StateGraph(AgentState)
    initial_state = assistantAgent({"question": question, "finishedAgents": set()})
    context = initial_state["question_type"]
    workflow.add_node("assistant_agent", lambda x: initial_state)
    workflow.add_edge(START, "assistant_agent")

    if "general_agent" in context:
        workflow.add_node("general_agent", generalAgent)
        workflow.add_edge("assistant_agent", "general_agent")
        workflow.add_edge("general_agent", "resultWriter_agent")
    if "travelguide_agent" in context:
        workflow.add_node("travelguide_agent", travelGuideAgent)
        workflow.add_node("regulation_agent", regulationAgent)
        workflow.add_edge("assistant_agent", "travelguide_agent")
        workflow.add_edge("travelguide_agent", "regulation_agent")
        workflow.add_edge("regulation_agent", "resultWriter_agent")
    if "travelplanner_agent" in context:
        workflow.add_node("travelplanner_agent", travelPlannerAgent)
        workflow.add_node("regulation_agent", regulationAgent)
        workflow.add_edge("assistant_agent", "travelplanner_agent")
        workflow.add_edge("travelplanner_agent", "regulation_agent")
        workflow.add_edge("regulation_agent", "resultWriter_agent")
    if "regulation_agent" in context:
        workflow.add_node("regulation_agent", regulationAgent)
        workflow.add_edge("assistant_agent", "regulation_agent")
        workflow.add_edge("regulation_agent", "resultWriter_agent")

    workflow.add_node("resultWriter_agent", resultWriterAgent)
    workflow.add_edge("resultWriter_agent", END)

    graph = workflow.compile()
    result = graph.invoke({"question": question})
    get_graph_image(graph)

    answers = result.get("responseFinal", [])
    contexts = result.get("answerAgents", "")
    return contexts, answers


def travelPlannerAgent(state: AgentState):
    print("\n--- TRAVELPLANNER AGENT ---")

    origin = state["origin"]
    destination = state["destination"]
    preference = state["preference"]

    prompt = f"""
        Anda adalah Travel Planner dalam Mlali Agents, yang memiliki pengetahuan yang sangat luas dan hebat hanya tentang perencanaan dalam perjalanan wisata.
        Tugas anda adalah memberikan perencanaan perjalanan dari {origin} ke {destination}, dengan preference: {preference}.
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_llm(messages)
    print("\n\nTRAVELPLANNER ANSWER:::", response)
    state["travelplannerResponse"] = response
    return {"travelplannerResponse": state["travelplannerResponse"]}


def regulationAgent(state: AgentState):
    print("\n--- REGULATION AGENT ---")

    prompt = f"""
        Anda adalah Travel Regulation dalam Mlali Agents, yang memiliki pengetahuan yang sangat luas dan hebat hanya tentang regulasi atau aturan-aturan pada tempat-tempat atau daerah wisata.
        Tugas anda adalah memberikan aturan-aturan regulasi pada suatu daerah atau tempat wisata kepada pengguna sesuai informasi yang dituju.
        Jangan mengubah isi dari informasi yang diberikan, cukup tambahkan regulasi atau aturan-aturan sesuai dengan informasi yang diberikan.
        Tuliskan regulasinya secara implisit pada deskripsinya.
        Berikut deskripsinya: {state["travelplannerResponse"]}
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_llm(messages)
    state["regulationResponse"] = response
    return {"regulationResponse": state["regulationResponse"]}


def runModelWithForm(origin, destination, preference):
    workflow = StateGraph(AgentState)

    workflow.add_node("travelplanner_agent", travelPlannerAgent)
    workflow.add_node("regulation_agent", regulationAgent)
    workflow.add_edge(START, "travelplanner_agent")
    workflow.add_edge("travelplanner_agent", "regulation_agent")
    workflow.add_edge("regulation_agent", END)

    graph = workflow.compile()
    result = graph.invoke({
        "origin": origin,
        "destination": destination,
        "preference": preference
    })
    get_graph_image(graph)

    answers = result.get("regulationResponse", [])
    contexts = result.get("answerAgents", "")
    return contexts, answers



# DEBUG QUESTION
# runModelWithForm("SGR", "KINTAMANI", "mencari glamping")
# runModel("saya dari buleleng, ingin glamping")