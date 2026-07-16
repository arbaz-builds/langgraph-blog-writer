"""Router node — classifies whether the topic needs research or can go straight to planning."""
from langchain_core.messages import SystemMessage, HumanMessage
from state import State, RouterStructured
from llms import router_llm

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
