#!/usr/bin/env python3
"""
Tests for utility functions.
"""

import pytest
from src.utils import add, subtract, multiply, divide


def test_add():
    """Test the add function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract():
    """Test the subtract function."""
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5
    assert subtract(-2, -3) == 1


def test_multiply():
    """Test the multiply function."""
    assert multiply(3, 4) == 12
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0


def test_divide():
    """Test the divide function."""
    assert divide(10, 2) == 5
    assert divide(-6, 2) == -3
    assert divide(7, 2) == 3.5


def test_divide_by_zero():
    """Test divide by zero raises ValueError."""
    with pytest.raises(ValueError):
        divide(5, 0)