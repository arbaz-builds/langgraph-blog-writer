from typing import Literal, Optional, List, Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator


class IntroDecision(BaseModel):
    reply: str = Field(
        description="A short, friendly message to the user. If decision is "
                     "'unclear', introduce this agent as a blog-writing "
                     "assistant and ask what topic they'd like a blog about. "
                     "If decision is 'blog_request', briefly confirm the "
                     "topic understood and that writing is starting."
    )
    decision: Literal["blog_request", "unclear"] = Field(
        description="'blog_request' if the user (across this conversation) "
                     "has given enough detail to start writing a blog. "
                     "'unclear' if the message is casual chat, a greeting, "
                     "or still missing key details like topic."
    )
    RefindTopic: Optional[str] = Field(
        default=None,
        description="A clean, polished topic description built from the "
                     "full conversation, ready to pass to the planning "
                     "stage. Only filled when decision is 'blog_request'; "
                     "left as None otherwise."
    )


class RouterStructured(BaseModel):
    reasoning: str
    route: Literal["research", "planning"]
    query: list[str] = Field(default_factory=list)


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.")
    bullets: List[str] = Field(..., min_length=3, max_length=5, description="3-5 concrete, non-overlapping subpoints to cover in this section.")
    target_words: int = Field(..., description="Target word count for this section (120-450).")
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(..., description="Use common_mistakes exactly once in the plan.")


class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict):
    memory: Annotated[List[BaseMessage], add_messages]
    topic: str
    router_decision: RouterStructured
    plan: Plan
    evidence: Optional[EvidencePack]
    sections: Annotated[List[str], operator.add]
