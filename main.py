import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from typing import Literal, Optional, List, Annotated
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_tavily import TavilySearch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import operator
import asyncio

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

router_llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="deepseek-ai/deepseek-v4-flash",
    temperature=0,
    timeout=15
)

general_LLM = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="openai/gpt-oss-120b",
    timeout=60
)

fallback_LLM = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="mistralai/mistral-nemotron",
    timeout=60
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
    topic: str
    router_decision: RouterStructured
    plan: Plan
    evidence: Optional[EvidencePack]
    sections: Annotated[List[str], operator.add]
    

system_prompt_Router = """You are a query router.

Classify the user request into exactly one route:

- research -> needs current, external, factual, social, or verifiable information.
- planning -> can be answered through reasoning, coding, writing, explanation, brainstorming, planning, or general knowledge.

If unsure, choose research.

Return ONLY valid JSON:

{
"reasoning": "under 15 words",
"route": "research" | "planning",
"query": []
}

Rules:

- No markdown or extra text.
- If route="planning", query must be [].
- If route="research", generate 2-8 unique search queries.
- Queries must be concise (3-10 words), search-friendly, and cover different angles.
- Avoid duplicates, questions, and conversational phrasing.
- Reasoning must briefly justify the route."""

async def router_node(state: State):
    try:
        user_prompt = str(state.get("topic", ""))
        structured_llm = router_llm.with_structured_output(RouterStructured, method="json_mode")
        router_messages = [
            SystemMessage(content=system_prompt_Router),
            HumanMessage(content=user_prompt)
        ]
        router_output = await structured_llm.ainvoke(router_messages)
        return {"router_decision": router_output}
    except Exception as e:
        print(f"[Router] Error encountered: {e}")
        return {
            "router_decision": RouterStructured(
                route="planning",
                reasoning="Fallback due to error",
                query=[]
            )
        }

def router_condition(state: State):
    user_prompt = state["router_decision"].route
    if user_prompt == "research":
        return "research"
    return "plan"

async def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    try:
        tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=max_results)
        response = await tool.ainvoke({"query": query})
        results = response.get("results", [])
        normalized: List[dict] = []
        for r in results:
            normalized.append({
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or "",
                "published_at": r.get("published_date"),
                "source": r.get("source"),
            })
        return normalized
    except Exception as e:
        print(f"[Tavily] Search failed for query '{query}': {e}")
        return []

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

async def research_node(state: State) -> dict:
    queries = state["router_decision"].query or []
    max_results = 6
    results_list = await asyncio.gather(*[_tavily_search(q, max_results=max_results) for q in queries])
    raw_results: List[dict] = []
    for results in results_list:
        raw_results.extend(results)
    if not raw_results:
        return {"evidence": None}

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ]
    try:
        extractor = general_LLM.with_structured_output(EvidencePack)
        pack = await extractor.ainvoke(messages)
    except Exception as e:
        print(f"[Research] general_LLM extraction failed: {e}")
        try:
            extractor = fallback_LLM.with_structured_output(EvidencePack)
            pack = await extractor.ainvoke(messages)
        except Exception as e2:
            print(f"[Research] fallback_LLM also failed, proceeding with no evidence: {e2}")
            return {"evidence": None}

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    return {"evidence": EvidencePack(evidence=list(dedup.values()))}

System_message_planner = """You are a blog planning agent.

Given a topic and optional research evidence, produce a structured blog plan.

Rules:
- blog_title: compelling, SEO-friendly, specific
- audience: who will read this (e.g. "developers", "business leaders")
- tone: writing style (e.g. "practical", "conversational", "authoritative")
- tasks: 5-7 sections covering intro → core content → conclusion
- Each task must have:
  - title: clear section name
  - goal: one sentence — what reader learns
  - bullets: 3-5 concrete, non-overlapping subpoints
  - target_words: 150-400 per section
  - section_type: exactly one of [intro, core, examples, checklist, common_mistakes, conclusion]
- Use 'common_mistakes' section type exactly once
- Do NOT use generic filler — every section must add value

EVIDENCE USAGE:
- If Evidence is provided, assign each bullet that maps to a specific evidence fact so the Worker can cite it later — phrase that bullet around the concrete fact (e.g. "X company's Y% reduction in Z"), not a vague restatement of it.
- If Evidence is empty or does not cover a section, write that section's bullets as conceptual/qualitative points — do not reference "evidence" or "sources" in the bullet text itself.
- Do not distribute the same evidence fact across multiple sections — each fact should be planned into exactly one section to avoid repetition."""

async def orchestrator(state: State) -> dict:
    research = state.get("evidence")
    messages = [
        SystemMessage(content=System_message_planner),
        HumanMessage(content=f"Topic: {state['topic']}:\n"
                             f"Evidence (ONLY use for fresh claims; may be empty):\n"
                             f"DATA{research}")
    ]
    try:
        planner = general_LLM.with_structured_output(Plan)
        plan = await planner.ainvoke(messages)
    except Exception as e:
        print(f"[Orchestrator] general_LLM failed: {e}")
        try:
            planner = fallback_LLM.with_structured_output(Plan)
            plan = await planner.ainvoke(messages)
        except Exception as e2:
            print(f"[Orchestrator] fallback_LLM also failed, using rule-based plan: {e2}")
            plan = Plan(
                blog_title=state["topic"],
                audience="general readers",
                tone="informative",
                tasks=[
                    Task(id=1, title="Introduction",
                         goal=f"Introduce the topic: {state['topic']}",
                         bullets=["What this topic covers", "Why it matters", "What the reader will learn"],
                         target_words=200, section_type="intro"),
                    Task(id=2, title="Core Concepts",
                         goal="Explain the main ideas behind the topic",
                         bullets=["Key concept 1", "Key concept 2", "Key concept 3"],
                         target_words=350, section_type="core"),
                    Task(id=3, title="Common Mistakes",
                         goal="Highlight pitfalls readers should avoid",
                         bullets=["Common mistake 1", "Common mistake 2", "Common mistake 3"],
                         target_words=250, section_type="common_mistakes"),
                    Task(id=4, title="Conclusion",
                         goal="Summarize the key takeaways",
                         bullets=["Recap main points", "Final thoughts"],
                         target_words=150, section_type="conclusion"),
                ]
            )
    return {"plan": plan}

def fanout(state: State):
    return [
        Send(
            "Worker",
            {"task": task, "topic": state["topic"], "plan": state["plan"], "evidence": state.get("evidence")},
        )
        for task in state["plan"].tasks
    ]

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.

Write ONE blog section in Markdown.

Requirements:

- Start with "## {Section Title}".
- Fulfill the Section Goal.
- Cover ALL Bullets in the given order.
- Stay close to the Target Words count (+-15%).
- Use short paragraphs, lists, and code blocks only when helpful.
- Be clear, practical, and engaging.
- Avoid fluff, repetition, hype, and generic AI phrasing.
- Do not write content outside this section.
- Output only the section Markdown.

STRICT EVIDENCE RULES (critical):
- Only state a specific number, statistic, percentage, date, company name, or named example if it appears explicitly in the Evidence provided.
- If Evidence is empty or does not cover a bullet, write that bullet qualitatively (e.g. "a growing share of jobs", "significant cost declines") instead of inventing a figure.
- Never invent company names, project names, case studies, or examples that are not present in Evidence.
- If you use a number from Evidence, it must match exactly (do not round, extrapolate, or combine numbers from different sources into a new invented figure).
- It is better to write a shorter, vaguer section than to fabricate a specific-sounding fact.

CITATION RULES:
- When you use a specific fact, number, or claim from Evidence, cite it inline immediately after the sentence using the format: (Source: [domain or short name](url)).
- Do not cite general knowledge or your own qualitative reasoning — only cite facts that came from Evidence.
- If the same source supports multiple sentences in a row, cite it once at the end of that group, not after every sentence.
- Never fabricate a URL. Only use URLs that appear verbatim in Evidence.
- If Evidence is empty, write the section with no citations at all."""

async def worker_node(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]
    evidence = payload["evidence"]
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    messages = [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Topic: {topic}\n\n"
            f"Section: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Bullets: {bullets_text}\n"
            f"Evidence: {evidence}\n"
        ))
    ]
    try:
        response = await general_LLM.ainvoke(messages)
        section_md = response.content.strip()
    except Exception as e:
        print(f"[Worker] general_LLM failed, falling back: {e}")
        response = await fallback_LLM.ainvoke(messages)
        section_md = response.content.strip()
    return {"sections": [section_md]}

blog_graph = StateGraph(State)
blog_graph.add_node("Router", router_node)
blog_graph.add_node("research", research_node)
blog_graph.add_node("planning", orchestrator)
blog_graph.add_node("Worker", worker_node)

blog_graph.add_edge(START, "Router")
blog_graph.add_conditional_edges("Router", router_condition, {"research": "research", "plan": "planning"})
blog_graph.add_edge("research", "planning")
blog_graph.add_conditional_edges("planning", fanout, ["Worker"])
blog_graph.add_edge("Worker", END)

compiled_blog_agent = blog_graph.compile()

fastapi_app = FastAPI(
    title="Blog Writer Agent API",
    description="Multi-agent LangGraph pipeline: Router → Research → Planning → Parallel Writers",
    version="1.0.0",
    contact={"name": "Mohammad Arbaz", "url": "https://github.com/arbaz-builds"},
)

class QueryRequest(BaseModel):
    query_text: str = Field(
        ...,
        min_length=15,
        max_length=200,
        description="The blog topic to write about",
        examples=["How AI agents are changing software development"],
    )

class BlogResponse(BaseModel):
    blog_title: str
    sections: List[str]

@fastapi_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    messages = [err["msg"] for err in errors]
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "message": "; ".join(messages)}
    )

@fastapi_app.post(
    "/Agent",
    summary="Generate a blog post",
    description="Takes a topic and returns a fully researched, structured blog post with title and sections.",
    response_description="Blog title and list of markdown sections",
    response_model=BlogResponse,
    tags=["Blog Generation"],
)
async def BlogAgent(request: QueryRequest):
    try:
        inputs = {"topic": request.query_text}
        result = await compiled_blog_agent.ainvoke(inputs)
        return {
            "blog_title": result["plan"].blog_title,
            "sections": result["sections"]
        }
    except Exception as e:
        print(f"[BlogAgent] Server-side error: {e}")
        raise HTTPException(status_code=500, detail="Internal error occurred, please try again later")
