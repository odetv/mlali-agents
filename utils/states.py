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
    travelGuideResponseKeyword: str
    travelplannerResponse: str
    travelplannerResponseKeyword: str
    totalAgents: int
    finishedAgents: Set[str]
    answerAgents: Annotated[Sequence[AnswerState], add]
    responseFinal: str
    origin: str
    destination: str
    preference: str