"""
AI Concepts Demo - API Package
==============================
Aggregates all API routers into a single router for the main app.
"""

from fastapi import APIRouter

from .config import router as config_router
from .llm import router as llm_router
from .embeddings import router as embeddings_router
from .rag import router as rag_router
from .agent import router as agent_router

# Re-export commonly used items for convenience
from .config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBED_MODEL,
    get_current_llm_model,
    get_current_embed_model,
    check_ollama_available,
    reset_ollama_status,
)
from .utils import (
    safe_eval,
    get_embedding,
    cosine_similarity,
    OllamaUnavailableError,
    SAFE_FUNCTIONS,
    SAFE_CONSTANTS,
)

# Aggregate all routers
router = APIRouter()
router.include_router(config_router)
router.include_router(llm_router)
router.include_router(embeddings_router)
router.include_router(rag_router)
router.include_router(agent_router)

__all__ = [
    "router",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBED_MODEL",
    "get_current_llm_model",
    "get_current_embed_model",
    "check_ollama_available",
    "reset_ollama_status",
    "safe_eval",
    "get_embedding",
    "cosine_similarity",
    "OllamaUnavailableError",
    "SAFE_FUNCTIONS",
    "SAFE_CONSTANTS",
]
