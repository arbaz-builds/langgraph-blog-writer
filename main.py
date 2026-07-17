"""FastAPI app — exposes the blog-writing graph over HTTP."""
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from state import Plan, EvidencePack, RouterStructured
from graph import compiled_blog_agent

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
            "sections": result["sections"],
            "plan": result.get("plan"),
            "evidence": result.get("evidence"),
            "router_decision": result.get("router_decision")
        }
    except Exception as e:
        print(f"[BlogAgent] Server-side error: {e}")
        raise HTTPException(status_code=500, detail="Internal error occurred, please try again later")
