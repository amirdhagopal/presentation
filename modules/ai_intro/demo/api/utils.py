"""
Utility functions for AI demo.
Includes safe math evaluator, embeddings, and similarity functions.
"""

import ast
import math
import operator
from typing import Optional

import numpy as np
import ollama

from .config import (
    check_ollama_available,
    reset_ollama_status,
    get_current_embed_model,
)

# =============================================================================
# SAFE MATH EVALUATOR
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
# EMBEDDINGS UTILITIES
# =============================================================================

class OllamaUnavailableError(Exception):
    """Raised when Ollama service is not available."""
    pass


def get_embedding(text: str, model: Optional[str] = None) -> np.ndarray:
    """Get embedding vector for text."""
    if not check_ollama_available():
        raise OllamaUnavailableError("Ollama is not running. Please start Ollama and try again.")
    
    embed_model = model or get_current_embed_model()
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
