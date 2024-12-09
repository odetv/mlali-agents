from typing_extensions import TypedDict, Annotated, Sequence, Set
from operator import add


class AnswerState(TypedDict):
    answer = None


class AgentState(TypedDict):
    question: str
    question_type: str
    generalQuestion: str
    travelguideQuestion: str
    travelplannerQuestion: str
    regulationQuestion: str
    travelguideResponse: str
    travelplannerResponse: str
    totalAgents: int
    finishedAgents: Set[str]
    answerAgents: Annotated[Sequence[AnswerState], add]
    responseFinal: str