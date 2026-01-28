# AI Presentation Modules

A modular presentation system for demonstrating AI concepts with interactive demos. Built with FastAPI and designed to work with local Ollama models.

## Overview

This project provides a unified launcher for running self-contained presentation modules. Each module includes:
- **Slides** — HTML-based presentation slides  
- **Demo** — Interactive API endpoints and web UI for hands-on demonstrations

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure Ollama is running with required models
ollama pull qwen3:8b
ollama pull nomic-embed-text:v1.5

# 3. Run a module
python run.py ai_intro

# 4. Open in browser
open http://localhost:8001
```

## Project Structure

```
presentation/
├── run.py              # Unified launcher for all modules
├── requirements.txt    # Python dependencies
└── modules/
    └── ai_intro/       # AI Introduction module
        ├── slides/     # Presentation slides (HTML/CSS/JS)
        └── demo/       # Interactive demos with API
            ├── api.py              # FastAPI routes
            ├── knowledge/          # RAG knowledge base
            ├── static/             # Web UI assets
            └── *_demo.py           # CLI demo scripts
```

## Available Modules

| Module | Description |
|--------|-------------|
| `ai_intro` | Introduction to AI concepts: LLM, Embeddings, RAG, and Agents |

## Usage

```bash
# List available modules
python run.py

# Run a specific module
python run.py <module_name>
```

The launcher automatically:
- Mounts slides at `/slides`
- Mounts demo static files at `/static`
- Includes API routes from `demo/api.py`
- Redirects root `/` to slides

## Requirements

- **Python 3.9+**
- **[Ollama](https://ollama.ai)** — Local LLM runtime
- Dependencies: `fastapi`, `uvicorn`, `ollama`, `numpy`

## Adding New Modules

Create a new directory under `modules/` with this structure:

```
modules/your_module/
├── slides/
│   └── index.html      # Entry point for slides
└── demo/
    ├── api.py          # FastAPI router (optional)
    └── static/         # Web assets (optional)
```

Then run: `python run.py your_module`

---

> Built with ❤️ for ThoughtWorks AI presentations
