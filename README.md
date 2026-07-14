# LangGraph Blog Writer — Multi-Agent Blog Generation System

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-teal)](https://fastapi.tiangolo.com)
[![Tests](https://github.com/arbaz-builds/langgraph-blog-writer/actions/workflows/tests.yml/badge.svg)](https://github.com/arbaz-builds/langgraph-blog-writer/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A multi-agent blog-generation pipeline built with LangGraph. Give it a topic — it decides whether the topic needs live research, plans a structured outline, then writes every section **in parallel** using LangGraph's `Send()` fan-out API.

**Live API:** `https://langgraph-blog-writer-pl4n.onrender.com` — interactive docs at `/docs`

```bash
curl -X POST "https://langgraph-blog-writer-pl4n.onrender.com/Agent" \
     -H "Content-Type: application/json" \
     -d '{"query_text": "What is Retrieval Augmented Generation?"}'
```

---

## Architecture

```
User topic
    │
    ▼
┌────────┐
│ Router │  DeepSeek V4 Flash — classifies: research or planning
└───┬────┘
    │
    ├── research ──▶ ┌───────────────┐
    │                │  research     │  Tavily: 2-8 parallel search queries
    │                │  node         │  → LLM synthesizes an EvidencePack
    │                └───────┬───────┘
    │                        │
    └── planning ────────────▼
                     ┌───────────────┐
                     │ orchestrator  │  GPT-OSS 120B builds a structured Plan:
                     │               │  title, audience, tone, 5-7 Tasks
                     └───────┬───────┘
                             │
                  Send() fan-out (one per Task)
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
            ┌────────┐  ┌────────┐  ┌────────┐
            │Worker 1│  │Worker 2│  │Worker N│   parallel section writers
            └───┬────┘  └───┬────┘  └───┬────┘
                 └───────────┴───────────┘
                             │
                             ▼
                    sections merged (operator.add)
                             │
                             ▼
                     Final blog response
```

## How it works

1. **Router** — a structured-output call to `deepseek-ai/deepseek-v4-flash` classifies the topic as `research` (needs current/external facts) or `planning` (answerable from the model's own knowledge).
2. **Research** *(if routed there)* — runs 2-8 Tavily search queries **concurrently** (`asyncio.gather`), then has an LLM deduplicate and structure the raw results into an `EvidencePack` (title, url, snippet, date per item).
3. **Orchestrator** — turns the topic (+ evidence, if any) into a `Plan`: blog title, audience, tone, and 5-7 `Task` objects, each with a goal, 3-5 bullets, a target word count, and a section type (exactly one `common_mistakes` section is required).
4. **Fan-out** — `Send()` dispatches every `Task` to a `Worker` node simultaneously — sections are written in parallel, not one after another.
5. **Worker** — writes one Markdown section per task, grounded strictly in the evidence it was given (see "Evidence discipline" below).
6. Sections are collected via `operator.add` into the final response.

## Evidence discipline

The worker prompt enforces some deliberate anti-hallucination rules:
- Numbers, dates, company names, or named examples may only appear if they're present verbatim in the `EvidencePack` — the model is instructed to write qualitatively ("a growing share of jobs") rather than invent a specific-sounding stat.
- Facts drawn from evidence are cited inline as `(Source: [name](url))`; URLs are never fabricated.
- If evidence is empty, the section is written with no citations at all rather than faking one.

## Resilience / fallback chains

Router, research synthesis, and planning each try `general_LLM` (`openai/gpt-oss-120b`) first and fall back to `fallback_LLM` (`mistralai/mistral-nemotron`) on failure, with a final rule-based 4-section plan if both LLMs fail during planning. Both models are served through NVIDIA's API, which is a **known limitation**: under heavy concurrent load the free-tier endpoint has been observed returning `429`/degraded-function errors, occasionally for both models on the same request.

> **Known gap:** the worker node's fallback call is not itself wrapped in a try/except — if `general_LLM` *and* `fallback_LLM` both fail for a given section, that exception currently propagates up to a generic 500 instead of degrading gracefully. Everywhere else (router, research, orchestrator) both LLM attempts are guarded.

## Testing & CI

The pipeline's pure control-flow logic is unit-tested with pytest, and every push/PR to `main` runs the suite via GitHub Actions:

- **`test_graph_logic.py`** — `router_condition` and `fanout` (no LLM calls, deterministic)
- **`test_schemas.py`** — Pydantic model validation (`Task`, `Plan`, `EvidencePack`, etc.)
- **`test_research_node.py`** — research node behavior with mocked search/LLM calls

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -v
```

CI runs against dummy API keys — the tests exercise routing/validation logic, not live LLM or search calls.

## Project structure

```
langgraph-blog-writer/
├── main.py                      # Graph definition + FastAPI app
├── tests/
│   ├── test_graph_logic.py
│   ├── test_research_node.py
│   └── test_schemas.py
├── .github/workflows/tests.yml  # CI: runs pytest on every push/PR
├── render.yaml                  # Render deployment config
├── requirements.txt
├── requirements-test.txt
└── .env.example
```

## Setup & installation

```bash
git clone https://github.com/arbaz-builds/langgraph-blog-writer.git
cd langgraph-blog-writer
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
uvicorn main:fastapi_app --host 0.0.0.0 --port 8000 --reload
```

## API reference

### `POST /Agent`

**Request**
```json
{
  "query_text": "What is Retrieval Augmented Generation?"
}
```
`query_text` must be 15-200 characters.

**Response**
```json
{
  "blog_title": "RAG Explained: How AI Systems Retrieve and Generate Accurate Answers",
  "sections": [
    "## Introduction\n\nRetrieval-Augmented Generation (RAG)...",
    "## How RAG Works\n\n...",
    "## Common Mistakes\n\n...",
    "## Conclusion\n\n..."
  ],
  "plan": { "blog_title": "...", "audience": "...", "tone": "...", "tasks": [ ] },
  "evidence": { "evidence": [ { "title": "...", "url": "...", "snippet": "..." } ] },
  "router_decision": { "reasoning": "...", "route": "research", "query": ["..."] }
}
```

The response includes the full `plan`, `evidence`, and `router_decision` alongside the generated `sections` — useful for debugging or displaying how the agent arrived at the final post, not just the post itself.

Interactive docs: `/docs` (Swagger UI).

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | ✅ | LLM inference (router, planner, workers) |
| `TAVILY_API_KEY` | ✅ | Web search for the research node |
| `NVIDIA_BASE_URL` | optional | Defaults to `https://integrate.api.nvidia.com/v1` |

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (`StateGraph`, `Send()` fan-out) |
| LLMs | NVIDIA API — `deepseek-v4-flash` (routing), `gpt-oss-120b` (primary), `mistral-nemotron` (fallback) |
| Web search | Tavily |
| API framework | FastAPI |
| Validation | Pydantic v2 |
| Testing | pytest + GitHub Actions |
| Deployment | Render |

## Related projects

- [langgraph-multi-agent](https://github.com/arbaz-builds/langgraph-multi-agent) — multi-agent chatbot with RAG, web search, and Python execution via MCP
- [fastmcp-python-repl-server](https://github.com/arbaz-builds/fastmcp-python-repl-server) — the MCP Python REPL server used by the multi-agent chatbot

## Author

**Mohammad Arbaz** — AI/LLM Engineer
[GitHub @arbaz-builds](https://github.com/arbaz-builds) · arwazrozi@gmail.com

## License

MIT — see [LICENSE](LICENSE).
