"""Research node — runs Tavily searches for the router's queries and extracts structured evidence."""
import asyncio
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from state import State, EvidencePack
from llms import general_LLM, fallback_LLM
import config


async def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    try:
        tool = TavilySearch(tavily_api_key=config.TAVILY_API_KEY, max_results=max_results)
        response = await tool.ainvoke({"query": query})
        results = response.get("results", [])
        normalized: List[dict] = []
        for r in results:
            normalized.append({
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or "",
                "published_at": r.get("published_date"),
                "source": r.get("source"),
            })
        return normalized
    except Exception as e:
        print(f"[Tavily] Search failed for query '{query}': {e}")
        return []


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""


async def research_node(state: State) -> dict:
    queries = state["router_decision"].query or []
    max_results = 6
    results_list = await asyncio.gather(*[_tavily_search(q, max_results=max_results) for q in queries])
    raw_results: List[dict] = []
    for results in results_list:
        raw_results.extend(results)
    if not raw_results:
        return {"evidence": None}

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ]
    try:
        extractor = general_LLM.with_structured_output(EvidencePack)
        pack = await extractor.ainvoke(messages)
    except Exception as e:
        print(f"[Research] general_LLM extraction failed: {e}")
        try:
            extractor = fallback_LLM.with_structured_output(EvidencePack)
            pack = await extractor.ainvoke(messages)
        except Exception as e2:
            print(f"[Research] fallback_LLM also failed, proceeding with no evidence: {e2}")
            return {"evidence": None}

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    return {"evidence": EvidencePack(evidence=list(dedup.values()))}
