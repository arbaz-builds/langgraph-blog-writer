"""Worker node — writes one blog section in Markdown, invoked in parallel per task via Send()."""
from langchain_core.messages import SystemMessage, HumanMessage
from llms import general_LLM, fallback_LLM

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
