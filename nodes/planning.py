"""Planning node — builds the structured blog Plan, and fans out one Send() per section to Worker."""
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from state import State, Plan, Task
from llms import general_LLM, fallback_LLM

System_message_planner = """You are an expert Blog Planning Agent that creates a complete content blueprint for a Writer Agent.

Objective

Generate a structured, SEO-aware, platform-appropriate blog plan from a Topic and optional Evidence.

Priority

1. Follow explicit user instructions.
2. Infer missing preferences from the Topic.
3. Apply platform and content-type rules.
4. Use defaults only when information is unavailable.

Inputs

- Topic
- Optional Evidence

Determine:

- platform
- audience
- audience_level (Beginner, Intermediate, Advanced)
- tone
- content_type
- target_length

Defaults:

- Length: 800–1500 words
- Platform: Long-form blog
- Tone: Best fit for the audience
- Sections: 5–7

Platform adaptation:

- LinkedIn: 4–5 concise, hook-driven, conversational sections.
- Medium/Blog: Detailed, SEO-focused, includes a checklist.
- Newsletter: Short, scannable, one key takeaway per section.

Adapt the structure to the requested content type (tutorial, opinion, case study, comparison, how-to, listicle, technical guide, etc.).

Output

Return exactly one valid JSON object with no markdown, comments, or additional text.

Fields:

- blog_title
- audience
- audience_level
- platform
- tone
- content_type
- estimated_total_words
- primary_keyword
- secondary_keywords (3–8)
- search_intent (Informational, Navigational, Commercial, or Transactional)
- meta_description (≤160 characters)
- slug
- tasks

Tasks

Generate 4–7 ordered sections.

Each task includes:

- title
- goal
- bullets (3–5)
- target_words
- section_type
- transition_to_next

Allowed section_type values:
intro, core, examples, checklist, common_mistakes, conclusion.

Use common_mistakes exactly once unless inappropriate (e.g. short LinkedIn content). Allocate most words to core sections.

Evidence

If Evidence is provided:

- Map each fact to exactly one bullet.
- Never reuse evidence.
- Phrase bullets around concrete facts for future citation.

Otherwise, create conceptual planning only without mentioning evidence or sources.

Quality

Every section must introduce new information, avoid repetition and filler, progress logically, provide actionable value, and transition naturally to the next section.

Do not invent unnecessary details. When information cannot be inferred, use the defaults.

The final plan must be engaging, SEO-aware, platform-appropriate, and easy for a Writer Agent to expand."""


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
