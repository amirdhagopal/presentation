#!/usr/bin/env python3
"""
Tests for AI Intro Demo API
===========================
Basic test suite for the demo API endpoints.

Run from module directory:
    python -m pytest tests/test_api.py -v

Or from project root:
    python -m pytest modules/ai_intro/demo/tests/test_api.py -v
"""

import math
import pytest
import numpy as np

# Import from api package
from ..api import (
    safe_eval,
    cosine_similarity,
    SAFE_FUNCTIONS,
    SAFE_CONSTANTS,
)


class TestSafeEval:
    """Tests for the safe_eval function."""
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert safe_eval("2 + 3") == 5
        assert safe_eval("10 - 4") == 6
        assert safe_eval("3 * 4") == 12
        assert safe_eval("15 / 3") == 5
        assert safe_eval("17 // 5") == 3
        assert safe_eval("17 % 5") == 2
        assert safe_eval("2 ** 3") == 8
    
    def test_unary_operators(self):
        """Test unary operators."""
        assert safe_eval("-5") == -5
        assert safe_eval("+5") == 5
        assert safe_eval("--5") == 5
    
    def test_constants(self):
        """Test mathematical constants."""
        assert safe_eval("pi") == math.pi
        assert safe_eval("e") == math.e
    
    def test_functions(self):
        """Test supported mathematical functions."""
        assert safe_eval("sqrt(16)") == 4
        assert safe_eval("abs(-5)") == 5
        assert abs(safe_eval("sin(0)")) < 1e-10
        assert safe_eval("cos(0)") == 1
        assert safe_eval("floor(3.7)") == 3
        assert safe_eval("ceil(3.2)") == 4
    
    def test_complex_expressions(self):
        """Test complex expressions."""
        assert safe_eval("2 + 3 * 4") == 14
        assert safe_eval("(2 + 3) * 4") == 20
        assert safe_eval("sqrt(144) + 25") == 37
        assert abs(safe_eval("pi * 2") - 6.283185307179586) < 1e-10
    
    def test_invalid_expressions(self):
        """Test that invalid expressions raise ValueError."""
        with pytest.raises(ValueError):
            safe_eval("import os")
        
        with pytest.raises(ValueError):
            safe_eval("__import__('os')")
        
        with pytest.raises(ValueError):
            safe_eval("open('file.txt')")
        
        with pytest.raises(ValueError):
            safe_eval("unknown_func(5)")


class TestCosineSimilarity:
    """Tests for cosine similarity function."""
    
    def test_identical_vectors(self):
        """Test that identical vectors have similarity 1."""
        a = np.array([1, 2, 3])
        assert cosine_similarity(a, a) == pytest.approx(1.0)
    
    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity -1."""
        a = np.array([1, 0, 0])
        b = np.array([-1, 0, 0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)
    
    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity 0."""
        a = np.array([1, 0])
        b = np.array([0, 1])
        assert cosine_similarity(a, b) == pytest.approx(0.0)
    
    def test_zero_vector(self):
        """Test that zero vectors return 0."""
        a = np.array([0, 0, 0])
        b = np.array([1, 2, 3])
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0
