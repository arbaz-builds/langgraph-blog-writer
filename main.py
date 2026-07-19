"""FastAPI app — exposes the blog-writing graph over HTTP, with Postgres-backed memory."""
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from state import Plan, EvidencePack, RouterStructured
from graph import blog_graph
import config

config.validate()

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
    thread_id: str = Field(
        default="1",
        description="Conversation/session ID — used to persist memory across requests.",
    )


class BlogResponse(BaseModel):
    blog_title: str
    sections: List[str]
    plan: Plan
    evidence: Optional[EvidencePack] = None
    router_decision: RouterStructured


@fastapi_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    messages = [err["msg"] for err in errors]
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "message": "; ".join(messages)}
    )


async def _invoke(query_text: str, thread_id: str = "1") -> dict:
    """Shared logic: fresh Postgres connection per call, run one turn,
    return the full result dict.

    Only 'memory' is passed in — intro_router reads the conversation from
    memory and sets 'topic' itself once it has enough detail. A new
    connection is opened each time (instead of one long-lived connection)
    so we never reuse a stale/dead TCP connection left over from a
    provider-side idle timeout (e.g. Neon auto-suspend, Render free-tier
    sleep) — same pattern as langgraph-multi-agent.
    """
    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
        await cp.setup()
        result = await blog_graph.compile(checkpointer=cp).ainvoke(
            {"memory": [HumanMessage(content=query_text)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    return result


@fastapi_app.post(
    "/Agent",
    summary="Generate a blog post",
    description="Takes a topic and returns a fully researched, structured blog post with title and sections, or a clarifying reply if more detail is needed.",
    response_description="Blog title and list of markdown sections, or a clarifying reply",
    tags=["Blog Generation"],
)
async def BlogAgent(request: QueryRequest):
    try:
        result = await _invoke(request.query_text, request.thread_id)
        if "plan" not in result:
            # intro_router decided this wasn't a blog request yet
            return {"reply": result["memory"][-1].content}
        return {
            "blog_title": result["plan"].blog_title,
            "sections": result["sections"],
            "plan": result.get("plan"),
            "evidence": result.get("evidence"),
            "router_decision": result.get("router_decision")
        }
    except Exception as e:
        print(f"[BlogAgent] Server-side error: {e}")
        raise HTTPException(status_code=500, detail="Internal error occurred, please try again later")
