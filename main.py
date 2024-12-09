from langgraph.graph import END, START, StateGraph
from utils.agents import assistantAgent, generalAgent, regulationAgent, resultWriterAgent, travelGuideAgent, travelPlannerAgent
from utils.create_graph_image import get_graph_image
from utils.states import AgentState


def run(question):
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

# DEBUG QUESTION
run("halo")