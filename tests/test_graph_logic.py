"""
Unit tests for pure graph-control-flow functions: router_condition and fanout.
These contain no LLM calls, so they run instantly and deterministically.
"""
from main import router_condition, fanout, RouterStructured, Plan, Task


class TestRouterCondition:
    def test_routes_to_research(self):
        state = {
            "router_decision": RouterStructured(
                reasoning="needs facts", route="research", query=["x"]
            )
        }
        assert router_condition(state) == "research"

    def test_routes_to_plan(self):
        state = {
            "router_decision": RouterStructured(
                reasoning="pure reasoning", route="planning", query=[]
            )
        }
        assert router_condition(state) == "plan"


class TestFanout:
    def _make_plan(self, n_tasks=3):
        tasks = [
            Task(
                id=i,
                title=f"Section {i}",
                goal="Explain something",
                bullets=["a", "b", "c"],
                target_words=200,
                section_type="core",
            )
            for i in range(n_tasks)
        ]
        return Plan(blog_title="Test Blog", audience="devs", tone="practical", tasks=tasks)

    def test_fanout_creates_one_send_per_task(self):
        plan = self._make_plan(n_tasks=4)
        state = {"topic": "RAG systems", "plan": plan, "evidence": None}
        sends = fanout(state)
        assert len(sends) == 4

    def test_fanout_send_payload_contains_expected_keys(self):
        plan = self._make_plan(n_tasks=1)
        state = {"topic": "RAG systems", "plan": plan, "evidence": None}
        sends = fanout(state)
        payload = sends[0].arg
        assert payload["topic"] == "RAG systems"
        assert payload["task"].id == 0
        assert payload["plan"] is plan
        assert payload["evidence"] is None

    def test_fanout_targets_worker_node(self):
        plan = self._make_plan(n_tasks=2)
        state = {"topic": "x", "plan": plan, "evidence": None}
        sends = fanout(state)
        assert all(s.node == "Worker" for s in sends)
