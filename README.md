# 🤖 LangGraph Blog Writer — Multi-Agent AI Blog Generation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-API-76B900.svg)](https://integrate.api.nvidia.com)
[![Tavily](https://img.shields.io/badge/Tavily-Search-orange.svg)](https://tavily.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-red.svg)](https://langgraph-blog-agent-1.onrender.com)

> A production-grade multi-agent blog generation system powered by LangGraph, NVIDIA LLMs, and real-time web research via Tavily. Give it a topic — it researches, plans, and writes a complete structured blog post using parallel AI workers.

---

## 🚀 Live Demo

**Try it now:** [https://langgraph-blog-agent-1.onrender.com/docs](https://langgraph-blog-agent-1.onrender.com/docs)

```bash
curl -X POST "https://langgraph-blog-agent-1.onrender.com/Agent" \
     -H "Content-Type: application/json" \
     -d '{"query_text": "What is Retrieval Augmented Generation?"}'
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────┐
│  Router │  ← DeepSeek V4 Flash — classifies: research or planning
└────┬────┘
     │
     ├──── research ────▶ ┌──────────────┐
     │                    │ Research Node │  ← Tavily multi-query web search
     │                    │              │     + LLM evidence synthesis
     │                    └──────┬───────┘
     │                           │
     └──── planning ─────────────▼
                          ┌──────────────┐
                          │ Orchestrator │  ← GPT-OSS 120B — creates structured blog plan
                          │              │     (5-7 sections with goals, bullets, word targets)
                          └──────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │  LangGraph Send() API   │
                    ▼            ▼            ▼
               ┌────────┐  ┌────────┐  ┌────────┐
               │Worker 1│  │Worker 2│  │Worker N│  ← Parallel section writers
               └────┬───┘  └────┬───┘  └────┬───┘
                    └───────────┴────────────┘
                                │
                                ▼
                        Final Blog Post
```

---

## ✨ Key Features

- **Intelligent Query Routing** — DeepSeek V4 Flash classifies each query: needs research or can be answered from knowledge alone
- **Real-Time Web Research** — Tavily runs 2–8 targeted search queries in parallel, synthesizes authoritative evidence
- **Structured Blog Planning** — GPT-OSS 120B generates a complete plan: title, audience, tone, 5–7 sections with goals and word targets
- **Parallel Section Writing** — LangGraph's `Send()` API fans out all sections to independent worker agents simultaneously — zero sequential bottleneck
- **Pydantic-Validated Outputs** — Every LLM output is schema-validated (`RouterStructured`, `Plan`, `Task`, `EvidencePack`) — no hallucinated structure
- **Production Deployed** — FastAPI + Render deployment, live and callable via REST API

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph (StateGraph, Send API) |
| **LLMs** | NVIDIA API — DeepSeek V4 Flash (routing), GPT-OSS 120B (planning + writing) |
| **Web Search** | Tavily Search API |
| **API Framework** | FastAPI |
| **Data Validation** | Pydantic v2 |
| **Deployment** | Render |

---

## 📁 Project Structure

```
langgraph-blog-writer/
├── main.py              # Full agent graph + FastAPI app
├── requirements.txt     # Dependencies
├── render.yaml          # Render deployment config
├── .env.example         # Environment variables template
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/arbaz-builds/langgraph-blog-writer.git
cd langgraph-blog-writer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Run the server

```bash
uvicorn main:fastapi_app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test it

```bash
curl -X POST "http://localhost:8000/Agent" \
     -H "Content-Type: application/json" \
     -d '{"query_text": "How does LangGraph work?"}'
```

---

## 📡 API Reference

### `POST /Agent`

Generate a complete blog post from a topic query.

**Request:**
```json
{
  "query_text": "What is Retrieval Augmented Generation?"
}
```

**Response:**
```json
{
  "blog_title": "RAG Explained: How AI Systems Retrieve and Generate Accurate Answers",
  "sections": [
    "## Introduction\n\nRetrieval-Augmented Generation (RAG)...",
    "## How RAG Works\n\n...",
    "## Real-World Applications\n\n...",
    "## Common Mistakes\n\n...",
    "## Conclusion\n\n..."
  ]
}
```

**Interactive docs:** `/docs` (Swagger UI)

---

## 🧠 How It Works — Step by Step

1. **User sends a topic** via POST `/Agent`
2. **Router Node** — DeepSeek classifies whether the topic needs real-time research or can be planned directly
3. **Research Node** (if needed) — Tavily executes multiple targeted searches, LLM synthesizes evidence into a clean `EvidencePack`
4. **Orchestrator Node** — GPT-OSS 120B creates a detailed `Plan` with blog title, audience, tone, and 5–7 structured `Task` objects
5. **Fan-out via Send()** — LangGraph dispatches all tasks to parallel `Worker` agents simultaneously
6. **Worker Nodes** — Each worker writes one complete blog section in Markdown, grounded in evidence
7. **State aggregation** — All sections are collected via `operator.add` into the final blog

---

## 🔑 Getting API Keys

| Service | Link |
|---|---|
| NVIDIA API | [integrate.api.nvidia.com](https://integrate.api.nvidia.com) |
| Tavily Search | [tavily.com](https://tavily.com) |

---

## 🤝 Related Projects

- [langgraph-multi-agent](https://github.com/arbaz-builds/langgraph-multi-agent) — Production multi-agent chatbot with RAG, web search, and Python execution via MCP
- [fastmcp-python-repl-server](https://github.com/arbaz-builds/fastmcp-python-repl-server) — Secure sandboxed Python REPL server implementing the Model Context Protocol

---

## 👤 Author

**Mohammad Arbaz** — AI/LLM Engineer
- GitHub: [@arbaz-builds](https://github.com/arbaz-builds)
- Email: arwazrozi@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
