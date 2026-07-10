"""
Unit tests for the Pydantic schemas used across the blog-writing graph.
These tests do NOT call any LLM or external API - they only validate that
the data models behave as expected (required fields, defaults, constraints).
"""
import pytest
from pydantic import ValidationError

from main import (
    RouterStructured,
    Task,
    Plan,
    EvidenceItem,
    EvidencePack,
)


class TestRouterStructured:
    def test_valid_research_route(self):
        r = RouterStructured(
            reasoning="Needs current info",
            route="research",
            query=["langgraph tutorial", "rag best practices"],
        )
        assert r.route == "research"
        assert len(r.query) == 2

    def test_valid_planning_route_with_empty_query(self):
        r = RouterStructured(reasoning="No research needed", route="planning", query=[])
        assert r.route == "planning"
        assert r.query == []

    def test_query_defaults_to_empty_list(self):
        r = RouterStructured(reasoning="test", route="planning")
        assert r.query == []

    def test_invalid_route_value_rejected(self):
        with pytest.raises(ValidationError):
            RouterStructured(reasoning="test", route="not_a_valid_route", query=[])


class TestTask:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            id=1,
            title="Introduction",
            goal="Introduce the topic clearly.",
            bullets=["point one", "point two", "point three"],
            target_words=200,
            section_type="intro",
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_task(self):
        t = Task(**self._base_kwargs())
        assert t.section_type == "intro"
        assert len(t.bullets) == 3

    def test_bullets_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            Task(**self._base_kwargs(bullets=["only one"]))

    def test_bullets_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            Task(**self._base_kwargs(bullets=["a", "b", "c", "d", "e", "f"]))

    def test_invalid_section_type_rejected(self):
        with pytest.raises(ValidationError):
            Task(**self._base_kwargs(section_type="not_a_real_type"))


class TestPlan:
    def test_valid_plan(self):
        task = Task(
            id=1,
            title="Intro",
            goal="Introduce",
            bullets=["a", "b", "c"],
            target_words=150,
            section_type="intro",
        )
        plan = Plan(
            blog_title="How RAG Works",
            audience="developers",
            tone="practical",
            tasks=[task],
        )
        assert plan.blog_title == "How RAG Works"
        assert len(plan.tasks) == 1

    def test_plan_requires_tasks_field(self):
        with pytest.raises(ValidationError):
            Plan(blog_title="X", audience="Y", tone="Z", tasks="not-a-list")


class TestEvidenceItemAndPack:
    def test_evidence_item_requires_title_and_url(self):
        with pytest.raises(ValidationError):
            EvidenceItem(url="https://example.com")  # missing title

    def test_evidence_item_optional_fields_default_none(self):
        item = EvidenceItem(title="Some Article", url="https://example.com")
        assert item.published_at is None
        assert item.snippet is None
        assert item.source is None

    def test_evidence_pack_defaults_to_empty_list(self):
        pack = EvidencePack()
        assert pack.evidence == []

    def test_evidence_pack_holds_items(self):
        item = EvidenceItem(title="A", url="https://a.com")
        pack = EvidencePack(evidence=[item])
        assert len(pack.evidence) == 1
        assert pack.evidence[0].title == "A"
