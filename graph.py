"""Graph assembly — wires Router -> Research -> Planning -> parallel Worker fan-out."""
from langgraph.graph import StateGraph, START, END
from state import State
from nodes import router_node, router_condition, research_node, orchestrator, fanout, worker_node

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
