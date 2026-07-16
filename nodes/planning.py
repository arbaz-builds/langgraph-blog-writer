"""Planning node — builds the structured blog Plan, and fans out one Send() per section to Worker."""
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from state import State, Plan, Task
from llms import general_LLM, fallback_LLM

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
