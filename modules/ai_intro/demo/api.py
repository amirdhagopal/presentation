#!/usr/bin/env python3
"""
AI Concepts Demo - Web API Module
=================================
FastAPI router for AI demo endpoints.
"""

import ast
import math
import operator
import os
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import ollama
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

# Create Router instead of App
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


# Note: Static mounting and CORS are handled by the main runner.

# =============================================================================
# SAFE MATH EVALUATOR (replaces eval)
# =============================================================================

# Supported operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported functions
SAFE_FUNCTIONS = {
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'log': math.log,
    'log10': math.log10,
    'abs': abs,
    'round': round,
    'floor': math.floor,
    'ceil': math.ceil,
}

# Supported constants
SAFE_CONSTANTS = {
    'pi': math.pi,
    'e': math.e,
}


def safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    Only supports basic math operations and common functions.
    """
    try:
        tree = ast.parse(expression, mode='eval')
        return _eval_node(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


def _eval_node(node):
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):  # Numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    elif isinstance(node, ast.Name):  # Constants like pi, e
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown constant: {node.id}")
    
    elif isinstance(node, ast.BinOp):  # Binary operations
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type.__name__}")
    
    elif isinstance(node, ast.UnaryOp):  # Unary operations
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    
    elif isinstance(node, ast.Call):  # Function calls
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            args = [_eval_node(arg) for arg in node.args]
            return SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"Unsupported function: {getattr(node.func, 'id', 'unknown')}")
    
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# =============================================================================
# CONFIG ENDPOINTS
# =============================================================================


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


# =============================================================================
# UTILITIES
# =============================================================================

class OllamaUnavailableError(Exception):
    """Raised when Ollama service is not available."""
    pass


def get_embedding(text: str, model: Optional[str] = None) -> np.ndarray:
    """Get embedding vector for text."""
    if not check_ollama_available():
        raise OllamaUnavailableError("Ollama is not running. Please start Ollama and try again.")
    
    embed_model = model or current_embed_model
    try:
        response = ollama.embed(model=embed_model, input=text)
        return np.array(response['embeddings'][0])
    except Exception as e:
        reset_ollama_status()  # Reset so next call will re-check
        raise OllamaUnavailableError(f"Failed to get embedding: {str(e)}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# LLM ENDPOINTS
# =============================================================================

@router.get("/api/llm/generate")
async def llm_generate(
    prompt: str = Query(..., description="The prompt to generate from"),
    temperature: float = Query(0.7, ge=0, le=2),
    model: Optional[str] = Query(None, description="Override LLM model"),
):
    """Stream LLM generation."""
    llm_model = model or current_llm_model
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            for chunk in ollama.generate(
                model=llm_model,
                prompt=prompt,
                stream=True,
                options={"temperature": temperature}
            ):
                yield f"data: {chunk['response']}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/api/llm/chat")
async def llm_chat(
    message: str = Query(...),
    system: str = Query("You are a helpful assistant."),
    model: Optional[str] = Query(None),
):
    """Stream chat completion."""
    llm_model = model or current_llm_model
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ]
            for chunk in ollama.chat(model=llm_model, messages=messages, stream=True):
                if 'message' in chunk and 'content' in chunk['message']:
                    yield f"data: {chunk['message']['content']}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")


# =============================================================================
# EMBEDDINGS ENDPOINTS
# =============================================================================

@router.get("/api/embeddings/compare")
async def embeddings_compare(
    text1: str = Query(...),
    text2: str = Query(...),
):
    """Compare similarity between two texts."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    similarity = cosine_similarity(emb1, emb2)
    
    return {
        "text1": text1,
        "text2": text2,
        "similarity": round(similarity, 4),
        "interpretation": "Very similar" if similarity > 0.7 else "Somewhat related" if similarity > 0.5 else "Different topics"
    }


@router.get("/api/embeddings/search")
async def embeddings_search(
    query: str = Query(...),
):
    """Search through sample documents."""
    documents = [
        "Python is a popular programming language for data science.",
        "Machine learning models learn patterns from data.",
        "The Eiffel Tower is located in Paris, France.",
        "Neural networks are inspired by the human brain.",
        "Coffee is one of the most consumed beverages worldwide.",
        "Deep learning is a subset of machine learning.",
        "Large Language Models understand and generate text.",
        "RAG retrieves relevant knowledge for AI systems.",
    ]
    
    query_emb = get_embedding(query)
    
    results = []
    for doc in documents:
        doc_emb = get_embedding(doc)
        sim = cosine_similarity(query_emb, doc_emb)
        results.append({"document": doc, "similarity": round(sim, 4)})
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"query": query, "results": results[:5]}


# =============================================================================
# RAG ENDPOINTS
# =============================================================================

# Load knowledge base
KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "sample_docs.txt"
KNOWLEDGE_CHUNKS = []
KNOWLEDGE_EMBEDDINGS = []


def load_knowledge():
    """Load and embed knowledge base."""
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_EMBEDDINGS
    
    if KNOWLEDGE_CHUNKS:
        return  # Already loaded
    
    if not KNOWLEDGE_PATH.exists():
        return
    
    text = KNOWLEDGE_PATH.read_text()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for para in paragraphs:
        if len(para) > 50:  # Skip very short paragraphs
            KNOWLEDGE_CHUNKS.append(para)
            KNOWLEDGE_EMBEDDINGS.append(get_embedding(para))


@router.get("/api/rag/query")
async def rag_query(
    question: str = Query(...),
    top_k: int = Query(2, ge=1, le=5),
):
    """Perform RAG query with streaming response."""
    load_knowledge()
    llm_model = current_llm_model
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            # Retrieve
            yield f"data: [STEP] Searching knowledge base...\n\n"
            
            query_emb = get_embedding(question)
            similarities = []
            for i, (chunk, emb) in enumerate(zip(KNOWLEDGE_CHUNKS, KNOWLEDGE_EMBEDDINGS)):
                sim = cosine_similarity(query_emb, emb)
                similarities.append((sim, chunk))
            
            similarities.sort(reverse=True)
            top_chunks = similarities[:top_k]
            
            for sim, chunk in top_chunks:
                preview = chunk[:100].replace('\n', ' ')
                yield f"data: [CONTEXT] [{sim:.3f}] {preview}...\n\n"
            
            # Augment + Generate
            yield f"data: [STEP] Generating answer...\n\n"
            
            context = "\n---\n".join([chunk for _, chunk in top_chunks])
            prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
            
            for chunk in ollama.generate(model=llm_model, prompt=prompt, stream=True, options={"temperature": 0.3}):
                yield f"data: {chunk['response']}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")


# =============================================================================
# AGENT ENDPOINTS
# =============================================================================

# Agent tools
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        result = safe_eval(expression)
        return f"Result: {result}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_current_time() -> str:
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def web_search(query: str) -> str:
    mock_results = {
        "weather": "Current weather: Sunny, 72°F. Partly cloudy afternoon.",
        "python": "Python 3.12 is the latest stable version.",
        "ai": "AI is transforming industries through automation and intelligence.",
    }
    for key, response in mock_results.items():
        if key in query.lower():
            return f"Search results: {response}"
    return f"No specific results for '{query}'."


TOOLS = {
    "calculator": {"fn": calculator, "desc": "Evaluate math expressions"},
    "get_current_time": {"fn": get_current_time, "desc": "Get current time"},
    "web_search": {"fn": web_search, "desc": "Search the web"},
}


def build_agent_prompt():
    tool_list = "\n".join([f"- {name}: {info['desc']}" for name, info in TOOLS.items()])
    return f"""You are a helpful AI assistant with tools.

Available tools:
{tool_list}

Format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: [input]

After observation, continue or respond with:
Thought: I have enough information.
Final Answer: [your response]
"""


@router.get("/api/agent/run")
async def agent_run(query: str = Query(...)):
    """Run agent with ReAct pattern, streaming steps."""
    llm_model = current_llm_model
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            system_prompt = build_agent_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            for step in range(5):
                yield f"data: [STEP] Step {step + 1}\n\n"
                
                response = ollama.chat(model=llm_model, messages=messages, options={"temperature": 0.2})
                content = response['message']['content']
                
                # Extract thought
                thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', content, re.DOTALL)
                if thought_match:
                    yield f"data: [THOUGHT] {thought_match.group(1).strip()}\n\n"
                
                # Check for final answer
                final_match = re.search(r'Final Answer:\s*(.+)', content, re.DOTALL)
                if final_match:
                    yield f"data: [ANSWER] {final_match.group(1).strip()}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # Extract and execute action
                action_match = re.search(r'Action:\s*(\w+)', content)
                input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', content)
                
                if action_match:
                    action = action_match.group(1).strip()
                    action_input = input_match.group(1).strip() if input_match else ""
                    
                    yield f"data: [ACTION] {action}({action_input})\n\n"
                    
                    if action in TOOLS:
                        if action == "get_current_time":
                            observation = TOOLS[action]["fn"]()
                        else:
                            observation = TOOLS[action]["fn"](action_input)
                        yield f"data: [OBSERVATION] {observation}\n\n"
                        
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"Observation: {observation}"})
                    else:
                        yield f"data: [ERROR] Unknown tool: {action}\n\n"
                        break
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "Please provide your final answer."})
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")


# =============================================================================
# HEALTH & INFO
# =============================================================================

@router.get("/api/health")
async def health():
    """Check API and model availability."""
    try:
        ollama.show(current_llm_model)
        ollama.show(current_embed_model)
        return {"status": "ok", "llm": current_llm_model, "embed": current_embed_model}
    except Exception as e:
        return {"status": "error", "message": str(e)}






