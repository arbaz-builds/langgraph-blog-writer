"""Intro router node — the entry point of the graph. Determines whether the
user wants a blog written or is just chatting, and builds a polished topic
once enough detail has been gathered across the conversation."""
from langchain_core.messages import SystemMessage, AIMessage
from state import State, IntroDecision
from llms import router_llm, fallback_LLM

INTRO_SYSTEM = """You are the entry point of a Blog Writing Agent. This agent
ONLY writes blogs — it cannot chat casually, answer unrelated questions, or
perform any other task.

Look at the conversation so far and decide:

- decision="blog_request" if the user has given enough detail (topic, and
  ideally type/length/tone) to start writing a blog.
- decision="unclear" if the message is casual chat, a greeting, or is still
  missing key details like the topic.

Always fill "reply":
- If "unclear": introduce yourself as a Blog Writing Agent and ask what
  topic they'd like a blog about (and any missing details like tone/length).
- If "blog_request": briefly confirm the topic you understood and say
  you're getting started.

If decision="blog_request", also fill "RefindTopic" with a clean, polished
topic description built from the conversation, ready to hand off to the
planning stage. Set it to null (not an empty string) if decision="unclear"."""


async def intro_router(state: State) -> dict:
    messages = [
        SystemMessage(content=INTRO_SYSTEM),
        *state["memory"][-6:],
    ]
    try:
        structured_llm = router_llm.with_structured_output(IntroDecision, method="function_calling")
        output = await structured_llm.ainvoke(messages)
    except Exception as e:
        print(f"[IntroRouter] router_llm failed, falling back: {e}")
        try:
            structured_llm = fallback_LLM.with_structured_output(IntroDecision, method="function_calling")
            output = await structured_llm.ainvoke(messages)
        except Exception as e2:
            print(f"[IntroRouter] fallback_LLM also failed: {e2}")
            output = None

    if output is None:
        # both LLMs failed — fallback response
        return {
            "memory": [AIMessage(content="I'm a Blog Writing Agent. What topic would you like a blog about?")],
            "topic": "",
            "decision": "unclear"
        }

    return {
        "memory": [AIMessage(content=output.reply)],
        "topic": output.RefindTopic,
        "decision": output.decision
    }


def intro_router_condition(state: State):
    decision = state.get("decision")
    if decision == "blog_request":
        return "router"
    return "END"
