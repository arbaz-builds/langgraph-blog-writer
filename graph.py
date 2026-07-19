"""Graph assembly — wires Intro -> Router -> Research -> Planning -> parallel Worker fan-out."""
from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    intro_router, intro_router_condition,
    router_node, router_condition,
    research_node, orchestrator, fanout, worker_node,
)

blog_graph = StateGraph(State)
blog_graph.add_node("intro_router", intro_router)
blog_graph.add_node("Router", router_node)
blog_graph.add_node("research", research_node)
blog_graph.add_node("planning", orchestrator)
blog_graph.add_node("Worker", worker_node)

blog_graph.add_edge(START, "intro_router")
blog_graph.add_conditional_edges("intro_router", intro_router_condition, {"router": "Router", "END": END})
blog_graph.add_conditional_edges("Router", router_condition, {"research": "research", "plan": "planning"})
blog_graph.add_edge("research", "planning")
blog_graph.add_conditional_edges("planning", fanout, ["Worker"])
blog_graph.add_edge("Worker", END)

compiled_blog_agent = blog_graph.compile()
