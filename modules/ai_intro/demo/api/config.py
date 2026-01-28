"""
Configuration and model management for AI demo.
"""

import os
from typing import Optional

import ollama
from fastapi import APIRouter, Query

router = APIRouter()

# Default Models (can be overridden via environment variables or API)
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:8b")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")

# Current active models (can be changed via /api/config)
current_llm_model = DEFAULT_LLM_MODEL
current_embed_model = DEFAULT_EMBED_MODEL

# Ollama connection status
_ollama_available = None


def check_ollama_available() -> bool:
    """Check if Ollama is available and cache the result."""
    global _ollama_available
    if _ollama_available is None:
        try:
            ollama.list()
            _ollama_available = True
        except Exception:
            _ollama_available = False
    return _ollama_available


def reset_ollama_status():
    """Reset Ollama status to force re-check."""
    global _ollama_available
    _ollama_available = None


def get_current_llm_model() -> str:
    """Get current LLM model name."""
    return current_llm_model


def get_current_embed_model() -> str:
    """Get current embedding model name."""
    return current_embed_model


@router.get("/api/models")
async def list_models():
    """List all available Ollama models."""
    try:
        models_response = ollama.list()
        models = []
        for model in models_response.get('models', []):
            name = model.get('name', model.get('model', ''))
            size = model.get('size', 0)
            # Categorize models
            is_embed = 'embed' in name.lower() or 'nomic' in name.lower()
            models.append({
                "name": name,
                "size_gb": round(size / (1024**3), 2) if size else 0,
                "type": "embedding" if is_embed else "llm"
            })
        return {
            "models": models,
            "current_llm": current_llm_model,
            "current_embed": current_embed_model
        }
    except Exception as e:
        return {"error": str(e), "models": []}


@router.post("/api/config")
async def set_config(
    llm_model: Optional[str] = Query(None),
    embed_model: Optional[str] = Query(None),
):
    """Update model configuration."""
    global current_llm_model, current_embed_model
    
    if llm_model:
        current_llm_model = llm_model
    if embed_model:
        current_embed_model = embed_model
    
    return {
        "llm_model": current_llm_model,
        "embed_model": current_embed_model
    }


@router.get("/api/config")
async def get_config():
    """Get current model configuration."""
    return {
        "llm_model": current_llm_model,
        "embed_model": current_embed_model,
        "defaults": {
            "llm_model": DEFAULT_LLM_MODEL,
            "embed_model": DEFAULT_EMBED_MODEL
        }
    }


@router.get("/api/health")
async def health():
    """Check API and model availability."""
    try:
        ollama.show(current_llm_model)
        ollama.show(current_embed_model)
        return {"status": "ok", "llm": current_llm_model, "embed": current_embed_model}
    except Exception as e:
        return {"status": "error", "message": str(e)}
