"""
Tests for research_node, with Tavily search and the LLM extractor mocked out
so no real API calls happen. These specifically guard against the
EvidencePack/None type-mismatch bug: research_node must always return
either an EvidencePack instance or None for the "evidence" key - never a
raw list.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from main import research_node, EvidencePack, EvidenceItem, RouterStructured


def _state_with_queries(queries):
    return {
        "topic": "Retrieval Augmented Generation",
        "router_decision": RouterStructured(
            reasoning="needs facts", route="research", query=queries
        ),
    }


@pytest.mark.asyncio
async def test_no_queries_returns_none_evidence():
    state = _state_with_queries([])
    result = await research_node(state)
    assert result["evidence"] is None


@pytest.mark.asyncio
async def test_no_search_results_returns_none_evidence():
    state = _state_with_queries(["some query"])
    with patch("main._tavily_search", new=AsyncMock(return_value=[])):
        result = await research_node(state)
    assert result["evidence"] is None


@pytest.mark.asyncio
async def test_successful_extraction_returns_evidencepack_not_list():
    state = _state_with_queries(["rag best practices"])
    fake_results = [
        {"title": "RAG Guide", "url": "https://example.com/rag", "snippet": "...", "published_at": None, "source": "example.com"}
    ]
    fake_pack = EvidencePack(
        evidence=[EvidenceItem(title="RAG Guide", url="https://example.com/rag")]
    )

    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(return_value=fake_pack)

    with patch("main._tavily_search", new=AsyncMock(return_value=fake_results)), \
         patch.object(type(__import__("main").general_LLM), "with_structured_output",
                       return_value=mock_bound_llm):
        result = await research_node(state)

    # The critical regression check: evidence must be an EvidencePack, never a plain list
    assert isinstance(result["evidence"], EvidencePack)
    assert not isinstance(result["evidence"], list)
    assert result["evidence"].evidence[0].url == "https://example.com/rag"


@pytest.mark.asyncio
async def test_deduplicates_by_url():
    state = _state_with_queries(["rag"])
    fake_results = [{"title": "A", "url": "https://x.com", "snippet": "", "published_at": None, "source": None}]
    dup_pack = EvidencePack(
        evidence=[
            EvidenceItem(title="A", url="https://x.com"),
            EvidenceItem(title="A duplicate", url="https://x.com"),
            EvidenceItem(title="B", url="https://y.com"),
        ]
    )
    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(return_value=dup_pack)

    with patch("main._tavily_search", new=AsyncMock(return_value=fake_results)), \
         patch.object(type(__import__("main").general_LLM), "with_structured_output",
                       return_value=mock_bound_llm):
        result = await research_node(state)

    assert isinstance(result["evidence"], EvidencePack)
    assert len(result["evidence"].evidence) == 2  # deduped by url
