"""Intro router node — the entry point of the graph. Determines whether the
user wants a blog written or is just chatting, and builds a polished topic
once enough detail has been gathered across the conversation."""
from langchain_core.messages import SystemMessage, AIMessage
from state import State, IntroDecision
from llms import router_llm

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
planning stage. Leave it empty if decision="unclear"."""


async def intro_router(state: State) -> dict:
    structured_llm = router_llm.with_structured_output(IntroDecision)
    messages = [
        SystemMessage(content=INTRO_SYSTEM),
        *state["memory"][-6:],
    ]
    output = await structured_llm.ainvoke(messages)

    return {
        "memory": [AIMessage(content=output.reply)],
        "topic": output.RefindTopic,
        "dicision": output.dicision
    }
