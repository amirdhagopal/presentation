# AI Concepts Demo

Interactive demonstrations of core AI concepts using local Ollama models.

![AI Concepts Demo UI](static/screenshot.png)

## Features

| Demo | Description |
|------|-------------|
| **LLM** | Text generation with temperature control and streaming |
| **Embeddings** | Semantic similarity comparison and document search |
| **RAG** | Retrieval-Augmented Generation with knowledge base |
| **Agent** | ReAct pattern with tool use (calculator, time, web search) |

## Prerequisites

- **Python 3.9+**
- **[Ollama](https://ollama.ai)** installed and running
- Required models:
  ```bash
  ollama pull qwen3:8b
  ollama pull nomic-embed-text:v1.5
  ```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start web UI
python -m uvicorn app:app --port 8001

# Open browser
open http://localhost:8001
```

## Project Structure

```
demo/
├── app.py              # FastAPI backend with SSE streaming
├── requirements.txt    # Python dependencies
├── static/
│   ├── index.html      # Web UI
│   ├── style.css       # Dark theme styling
│   └── app.js          # Frontend logic
├── knowledge/
│   └── sample_docs.txt # Knowledge base for RAG
└── CLI demos:
    ├── 1_llm_basics.py
    ├── 2_embeddings.py
    ├── 3_rag_demo.py
    └── 4_agent_demo.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/llm/generate` | GET | Stream LLM text generation |
| `/api/llm/chat` | GET | Chat with system message |
| `/api/embeddings/compare` | GET | Compare text similarity |
| `/api/embeddings/search` | GET | Semantic search documents |
| `/api/rag/query` | GET | RAG query with streaming |
| `/api/agent/run` | GET | Run agent with ReAct loop |
| `/api/health` | GET | Check API/model status |

## CLI Demos

Run standalone Python scripts:

```bash
python 1_llm_basics.py      # LLM fundamentals
python 2_embeddings.py      # Text embeddings
python 3_rag_demo.py        # RAG pipeline
python 4_agent_demo.py      # Agent with tools
```

## Models

| Purpose | Model |
|---------|-------|
| LLM | `qwen3:8b` |
| Embeddings | `nomic-embed-text:v1.5` |

## Architecture

```
┌─────────────┐     SSE      ┌─────────────┐
│  Frontend   │ ←──────────→ │  FastAPI    │
│  (HTML/JS)  │              │  Backend    │
└─────────────┘              └──────┬──────┘
                                    │
                             ┌──────▼──────┐
                             │   Ollama    │
                             │  (local)    │
                             └─────────────┘
```

---

> **Note:** MCP is demonstrated conceptually in the presentation slides.
