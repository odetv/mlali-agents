from langgraph.graph import END, START, StateGraph
from utils.agents import assistantAgent, generalAgent, regulationAgent, travelGuideAgent
from utils.agents_form import regulationFormAgent, travelPlannerFormAgent
from utils.create_graph_image import get_graph_image
from utils.debug_time import time_check
from utils.states import AgentState


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
        workflow.add_edge("general_agent", END)
    if "travelguide_agent" in context:
        workflow.add_node("travelguide_agent", travelGuideAgent)
        workflow.add_node("regulation_agent", regulationAgent)
        workflow.add_edge("assistant_agent", "travelguide_agent")
        workflow.add_edge("travelguide_agent", "regulation_agent")
        workflow.add_edge("regulation_agent", END)
    if "regulation_agent" in context:
        workflow.add_node("regulation_agent", regulationAgent)
        workflow.add_edge("assistant_agent", "regulation_agent")
        workflow.add_edge("regulation_agent", END)

    graph = workflow.compile()
    result = graph.invoke({"question": question})
    get_graph_image(graph)

    answers = result.get("responseFinal", [])
    contexts = result.get("answerAgents", "")
    return contexts, answers


def runModelWithForm(origin, destination, preference):
    workflow = StateGraph(AgentState)
    workflow.add_node("travelplannerform_agent", travelPlannerFormAgent)
    workflow.add_node("regulationform_agent", regulationFormAgent)
    
    workflow.add_edge(START, "travelplannerform_agent")
    workflow.add_edge("travelplannerform_agent", "regulationform_agent")
    workflow.add_edge("regulationform_agent", END)

    graph = workflow.compile()
    result = graph.invoke({
        "origin": origin,
        "destination": destination,
        "preference": preference
    })
    get_graph_image(graph)

    answers = result.get("responseFinal", [])
    contexts = result.get("answerAgents", "")
    return contexts, answers



# DEBUG QUESTION
# runModelWithForm("SINGARAJA", "DASONG PANCASARI", "mencari glamping")
# runModel("saya dari buleleng, ingin wisata ke pura uluwatu")